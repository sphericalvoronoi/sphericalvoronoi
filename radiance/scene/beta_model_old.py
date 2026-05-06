#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.compress_utils import compress_png, decompress_png, sort_param_dict
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
from gsplat.rendering import rasterization
import json
import time
from spherical_voronoi import spherical_voronoi

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False

try:
    from nsv_cuda import (
        nsv_radiance_cuda, nsv_sum_radiance_cuda, nsv_per_lobe_tau_sum_cuda,
        nsv_sum_rgb_sigmoid_cuda, nsv_sum_rgb_relu_cuda,
        # nsv_scalar_tau_cuda, nsv_scalar_tau_sum_cuda,
        # nsv_per_lobe_tau_cuda
    )
    HAS_NSV_CUDA = True
except ImportError:
    HAS_NSV_CUDA = False

_NSV_VARIANTS = ('wnsv_norm', 'nsv', 'wnsv_scalar')
# raw init so that F.softplus(x) == 1.0
_SOFTPLUS_INV_1 = float(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))


def _sh_dir_encode(dirs: torch.Tensor, sh_degree: int) -> torch.Tensor:
    """Compute real SH basis values up to sh_degree for direction vectors [N,3]."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    bands = [0.2821 * torch.ones_like(x)]  # l=0
    if sh_degree >= 1:
        bands += [0.4886 * y, 0.4886 * z, 0.4886 * x]  # l=1
    if sh_degree >= 2:
        bands += [
            1.0925 * x * y,
            1.0925 * y * z,
            0.3153 * (2*z*z - x*x - y*y),
            1.0925 * x * z,
            0.5463 * (x*x - y*y),
        ]  # l=2
    if sh_degree >= 3:
        bands += [
            0.5900 * y * (3*x*x - y*y),
            2.8906 * x * y * z,
            0.4572 * y * (4*z*z - x*x - y*y),
            0.3732 * z * (2*z*z - 3*x*x - 3*y*y),
            0.4572 * x * (4*z*z - x*x - y*y),
            1.4453 * z * (x*x - y*y),
            0.5900 * x * (x*x - 3*y*y),
        ]  # l=3
    return torch.stack(bands, dim=-1)


def _nsv_sum_weighted_feat(sites, features, directions):
    """Shared wnsv_norm + softmax-weighted sum (torch). Returns [N,3] pre-activation."""
    sites      = sites.contiguous().float()
    features   = features.contiguous().float()
    directions = directions.contiguous().float()
    tau2    = (sites * sites).sum(dim=-1)
    tau     = torch.sqrt(tau2 + 1e-12)
    sites_n = sites / tau.unsqueeze(-1)
    diff    = sites_n - directions.unsqueeze(1)
    dist    = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
    logits  = -tau * dist
    W       = torch.softmax(logits, dim=-1).unsqueeze(-1)
    return (W * features).sum(dim=1)


def nsv_sum_rgb_sigmoid_torch(sites, features, directions):
    """Native-PyTorch fused wnsv_norm + sum-aggregate + sigmoid (RGB, C=3).

    Mirrors `nsv_sum_rgb_sigmoid_cuda`. Inference-only — call under no_grad.
    """
    return torch.sigmoid(_nsv_sum_weighted_feat(sites, features, directions))


def nsv_sum_rgb_relu_torch(sites, features, directions):
    """Native-PyTorch fused wnsv_norm + sum-aggregate + relu (RGB, C=3).

    Mirrors `nsv_sum_rgb_relu_cuda`. Inference-only — call under no_grad.
    """
    return torch.relu(_nsv_sum_weighted_feat(sites, features, directions))


class VoronoiMLP_Torch(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128,
                 use_dir_enc: bool = False, sh_degree: int = 2):
        super().__init__()
        self.use_dir_enc = use_dir_enc
        self.sh_degree = sh_degree
        dir_dim = (sh_degree + 1) ** 2 if use_dir_enc else 0
        self.net = nn.Sequential(
            nn.Linear(in_features + dir_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, dirs: torch.Tensor = None) -> torch.Tensor:
        if self.use_dir_enc and dirs is not None:
            enc = _sh_dir_encode(dirs, self.sh_degree)
            x = torch.cat([x, enc], dim=-1)
        return self.net(x)


class VoronoiMLP_TCNN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128,
                 use_dir_enc: bool = False, sh_degree: int = 2):
        super().__init__()
        if not HAS_TCNN:
            raise ImportError("tinycudann not installed.")
        self.use_dir_enc = use_dir_enc
        self.sh_degree = sh_degree
        dir_dim = 0
        if use_dir_enc:
            # TCNN SphericalHarmonics: degree=sh_degree+1 gives (sh_degree+1)^2 outputs
            self._dir_enc = tcnn.Encoding(3, {"otype": "SphericalHarmonics", "degree": sh_degree + 1})
            dir_dim = (sh_degree + 1) ** 2
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": hidden,
            "n_hidden_layers": 1,
        }
        self.net = tcnn.Network(in_features + dir_dim, 3, network_config)

    def forward(self, x: torch.Tensor, dirs: torch.Tensor = None) -> torch.Tensor:
        if self.use_dir_enc and dirs is not None:
            # TCNN SH encoding expects directions mapped to [0, 1]
            dirs_01 = (dirs * 0.5 + 0.5).clamp(0.0, 1.0)
            enc = self._dir_enc(dirs_01).float()
            x = torch.cat([x, enc], dim=-1)
        shape = x.shape
        if len(shape) > 2:
            x_flat = x.reshape(-1, shape[-1])
            out = self.net(x_flat)
            return out.reshape(*shape[:-1], 3)
        return self.net(x)


VoronoiMLP = VoronoiMLP_TCNN if HAS_TCNN else VoronoiMLP_Torch


def get_voronoi_mlp(in_features: int, hidden: int = 128, backend: str = "tcnn",
                    use_dir_enc: bool = False, sh_degree: int = 2):
    if backend == "tcnn":
        if not HAS_TCNN:
            print("WARNING: tinycudann not available, falling back to torch")
            return VoronoiMLP_Torch(in_features, hidden, use_dir_enc=use_dir_enc, sh_degree=sh_degree)
        return VoronoiMLP_TCNN(in_features, hidden, use_dir_enc=use_dir_enc, sh_degree=sh_degree)
    elif backend == "torch":
        return VoronoiMLP_Torch(in_features, hidden, use_dir_enc=use_dir_enc, sh_degree=sh_degree)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'tcnn' or 'torch'.")


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def fibonacci_sphere(N):
    indices = torch.arange(0, N, dtype=torch.float32) + 0.5
    phi = math.pi * (3. - math.sqrt(5.))
    y = 1 - (indices / N) * 2
    radius = torch.sqrt(1 - y ** 2)
    theta = phi * indices
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    return torch.stack([x, y, z], dim=1)


class BetaModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def sb_params_activation(sb_params):
            softplus_sb_params = F.softplus(sb_params[..., :3], beta=math.log(2) * 10)
            sb_params = torch.cat([softplus_sb_params, sb_params[..., 3:]], dim=-1)
            return sb_params

        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.sb_params_activation = sb_params_activation
        self.beta_activation = beta_activation
        self.sites_activation = torch.nn.functional.normalize


    def __init__(self, color_representation='nsv', num_lobes=7, num_features=4,
                 mlp_backend='tcnn', forward_shading=False, nsv_aggregate='concat',
                 nsv_top_k=0, use_mlp_ema=False, use_dir_enc=False, sh_degree_dir=2,
                 feature_activation='sigmoid'):
        """
        Args:
            color_representation: 'voronoi', 'nsv', 'sh', 'sb', 'sgs'
            num_lobes: Number of Voronoi lobes
            num_features: Features per lobe (for 'nsv')
            mlp_backend: 'tcnn' or 'torch'
            forward_shading: MLP per-Gaussian (True) or deferred per-pixel (False). Only 'nsv'.
            nsv_aggregate: 'concat' → [N, K*C], 'sum' → [N, C]. Only 'nsv'.
            use_dir_enc: concatenate SH-encoded view direction to MLP input.
            sh_degree_dir: max SH degree for direction encoding (default 2 → 9 coefficients).
            feature_activation: final activation on raw features when no MLP ('sigmoid' or 'relu').
        """
        self.active_sh_degree = 0
        self.max_sh_degree = 3 if color_representation == 'sh' else 0
        self.sb_number = 2 if color_representation == 'sb' else 0
        self._xyz = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._shN = torch.empty(0)
        self._sb_params = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.background = torch.empty(0)
        self.training = True
        self.optimizer = None
        self.feat_optimizer = None
        self.tau_optimizer = None
        self.mlp_optimizer = None
        self.mlp_scheduler = None
        self.spatial_lr_scale = 0
        self.color_representation = color_representation
        self.mlp_backend = mlp_backend
        self.forward_shading = forward_shading
        self.nsv_aggregate = nsv_aggregate
        self.feature_activation = feature_activation
        self.nsv_top_k = int(nsv_top_k)
        self.use_mlp_ema = use_mlp_ema
        self.use_dir_enc = use_dir_enc
        self.sh_degree_dir = sh_degree_dir
        if color_representation == 'voronoi':
            self.max_sh_degree = 0
            self.sb_number = 0
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self._log_tau_sites = torch.empty(0)  # [N, K] per-site temperature
            self._sites_mask = None
            self.num_lobes = num_lobes

        if color_representation in _NSV_VARIANTS:
            self.max_sh_degree = 0
            self.sb_number = 0
            self._sites = torch.empty(0)
            self._features = torch.empty(0)
            self._sites_mask = None
            self.num_lobes = num_lobes
            self.num_features = num_features
            eff_k = self.nsv_top_k if 0 < self.nsv_top_k < num_lobes else num_lobes
            mlp_in = eff_k * num_features if nsv_aggregate == 'concat' else num_features
            self.mlp = None if num_features == 3 else get_voronoi_mlp(
                mlp_in, backend=mlp_backend, use_dir_enc=use_dir_enc, sh_degree=sh_degree_dir).cuda()
            self.mlp_ema = None
            self.feat_lr_init = 0.0
            self.mlp_lr_init = 0.0
            self.total_iterations = None  # set by training_setup
            self.feat_optimizer = None
            self.tau_optimizer = None
            # nsv: single global temperature scalar (softplus activated)
            if color_representation == 'nsv':
                self._log_tau = torch.empty(0)
            # wnsv_scalar: per-site-per-gaussian temperature scalars (softplus activated)
            if color_representation == 'wnsv_scalar':
                self._log_temps = torch.empty(0)

        if color_representation == 'sgs':
            self.max_sh_degree = 0
            self.sb_number = 0
            self._mu = torch.empty(0)
            self._alpha = torch.empty(0)
            self._lambda = torch.empty(0)
            self._dc = torch.empty(0)
            self.num_lobes = num_lobes
        self.setup_functions()

    @torch.no_grad()
    def capture(self, sparse=True):
        mask = self._sites_mask.bool()
        sites = self._sites[mask]
        lambds = self._lambd[mask]
        return (
            self._xyz,
            self._sb_params,
            self._scaling,
            self._rotation,
            self._opacity,
            self._beta,
            sites,
            lambds,
            mask,
            self.max_degree,
            self.spatial_lr_scale,
        )
        
    @torch.no_grad
    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._sb_params,
            self._scaling,
            self._rotation,
            self._opacity,
            self._beta,
            sites,
            lambds,
            self._sites_mask,
            self.max_degree,
            self.spatial_lr_scale,
        ) = model_args
        sites_full = torch.zeros((self._sites_mask.shape[0], self.max_degree, 3), device='cuda')
        lambd_full = torch.zeros_like(sites_full)
        sites_full[self._sites_mask] = sites
        lambd_full[self._sites_mask] = lambds
        self._sites = nn.Parameter(sites_full)
        self._lambd = nn.Parameter(lambd_full)
        self._sites_mask = self._sites_mask.bool()            
        self.training_setup(training_args)

    def restore_legacy(self, model_args, training_args):
        (
            self._xyz,
            self._sb_params,
            self._scaling,
            self._rotation,
            self._opacity,
            self._beta,
            self._sites,
            self._lambd,
            self._mask,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_shs(self):
        return torch.cat((self._sh0, self._shN), dim=1)

    @property
    def get_beta(self):
        return self.beta_activation(self._beta)

    @property
    def get_sb_params(self):
        return self.sb_params_activation(self._sb_params)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_color(self):
        if self.color_representation == 'voronoi':
            return self._colors
        elif self.color_representation == 'nsv':
            return self._features
        return None
    
    @property
    def get_tau(self):
        if self.color_representation == 'nsv':
            return F.softplus(self._log_tau)           # scalar → broadcasts to [N, K]
        if self.color_representation == 'wnsv_scalar':
            return F.softplus(self._log_temps)         # [N, K]
        # wnsv_norm: temperature = ||site||
        return torch.norm(self._sites, dim=-1) if self.training else self._tau
        
    @property
    def get_sites(self):
        return F.normalize(self._sites, dim=-1) if self.training else self._nn_sites
    
    @property
    def get_mu(self):
        return F.normalize(self._mu, dim=-1)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )
    
    @torch.no_grad()
    def eval(self):
        self.training = False
        if self.color_representation in ('voronoi',) + _NSV_VARIANTS:
            self._nn_sites = F.normalize(self._sites, dim=-1)
            if self._sites_mask is not None:
                self._nn_sites[~self._sites_mask.bool()] = 1e8
            # cached norm-based tau used only by wnsv_norm
            self._tau = torch.norm(self._sites, dim=-1)
        if self.color_representation == 'voronoi':
            self._nn_sites = F.normalize(self._sites, dim=-1)
            if self._sites_mask is not None:
                self._nn_sites[~self._sites_mask.bool()] = 1e8
            self._nn_tau = torch.exp(self._log_tau_sites).detach()
        if self.color_representation in _NSV_VARIANTS and self.mlp is not None:
            self.mlp.eval()

    def train(self):
        self.training = True
        if self.color_representation in _NSV_VARIANTS and self.mlp is not None:
            self.mlp.train()

    def update_mlp_ema(self, gamma: float = 0.95):
        if self.mlp_ema is None:
            return
        with torch.no_grad():
            for p_ema, p in zip(self.mlp_ema.parameters(), self.mlp.parameters()):
                p_ema.data.mul_(gamma).add_(p.data, alpha=1.0 - gamma)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def compute_voronoi_color(self, directions: torch.Tensor) -> torch.Tensor:
        if self.training:
            sites_n = F.normalize(self._sites, dim=-1)
            tau     = torch.exp(self._log_tau_sites)
        else:
            sites_n = self._nn_sites   # pre-cached in eval()
            tau     = self._nn_tau     # pre-cached in eval()
        return spherical_voronoi(directions.reshape(-1, 3), sites_n, tau, self._colors)

    def compute_radiance(self, directions, metric="l2"):
        if self.color_representation == "voronoi":
            return self.compute_voronoi_color(directions.reshape(-1, 3))

        if self.color_representation in _NSV_VARIANTS:
            top_k = getattr(self, "nsv_top_k", 0)
            use_topk = 0 < top_k < self.num_lobes

            # CUDA fast paths (no topk)
            if HAS_NSV_CUDA and not use_topk:
                sites_n = self.get_sites  # normalized [N, K, 3]
                if self.color_representation == 'wnsv_norm':
                    if self.nsv_aggregate == 'concat':
                        return nsv_radiance_cuda(self._sites, self._features, directions)
                    else:
                        return nsv_sum_radiance_cuda(self._sites, self._features, directions)
                elif self.color_representation == 'wnsv_scalar':
                    taus = self.get_tau  # [N, K]
                    if self.nsv_aggregate == 'concat':
                        return nsv_per_lobe_tau_cuda(sites_n, self._features, directions, taus)
                    else:
                        return nsv_per_lobe_tau_sum_cuda(sites_n, self._features, directions, taus)
                elif self.color_representation == 'nsv':
                    tau = self.get_tau  # scalar tensor
                    if self.nsv_aggregate == 'concat':
                        return nsv_scalar_tau_cuda(sites_n, self._features, directions, tau)
                    else:
                        return nsv_scalar_tau_sum_cuda(sites_n, self._features, directions, tau)

            # Python fallback: wnsv_norm without CUDA, or any variant with topk
            directions = directions.reshape(-1, 1, 3)
            sites_dir = self.get_sites   # [N, K, 3]
            tau = self.get_tau           # scalar (nsv), [N,K] (wnsv_norm/wnsv_scalar)

            dist = torch.norm(sites_dir - directions, dim=-1)  # [N, K]
            logits = -tau * dist                                # [N, K]

            if use_topk:
                topk_logits, topk_idx = torch.topk(logits, k=top_k, dim=-1)
                C = self._features.shape[-1]
                feat_k = torch.gather(
                    self._features, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C)
                )
                W = torch.softmax(topk_logits, dim=-1).unsqueeze(-1)
                weighted = W * feat_k
            else:
                W = torch.softmax(logits, dim=-1).unsqueeze(-1)
                weighted = W * self._features

            if self.nsv_aggregate == 'sum':
                V = weighted.sum(dim=1)
            else:
                V = weighted.reshape(weighted.shape[0], -1)
            return V
    
        if self.color_representation == 'sgs':
            directions = directions.reshape(-1, 1, 3)
            mu = self.get_mu
            dot_product = (mu * directions).sum(dim=-1).unsqueeze(-1)
            sharpness = F.softplus(self._sharpness)
            V = self._alpha * torch.exp(sharpness * (dot_product - 1)) 
            V = V.sum(dim=1)
            return torch.clamp_min(V, 0.0)
        
    def update_sites_mask(self):
        if self.color_representation == 'voronoi':
            self._sites_mask = self._colors.abs().sum(dim=-1) > 0
        elif self.color_representation in _NSV_VARIANTS:
            self._sites_mask = self._features.abs().sum(dim=-1) > 0

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        shs = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        shs[:, :3, 0] = fused_color
        shs[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = (
            knn(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[:, 1:] ** 2
        ).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.5 * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        betas = torch.zeros_like(opacities)

        sb_params = torch.zeros(
            (fused_point_cloud.shape[0], self.sb_number, 6), device="cuda"
        )
        theta = torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)
        phi = 2 * torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)
        sb_params[:, :, 3] = theta
        sb_params[:, :, 4] = phi

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._sh0 = nn.Parameter(
            shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._shN = nn.Parameter(
            shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._sb_params = nn.Parameter(sb_params.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))
        
        if self.color_representation == 'voronoi':
            N = self._xyz.shape[0]
            vectors = fibonacci_sphere(self.num_lobes)
            vectors = vectors.repeat(N, 1, 1).to('cuda')
            self._sites = nn.Parameter(vectors)
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda() - 0.5
            self._colors = nn.Parameter(colors.repeat(1, self.num_lobes).reshape(N, -1, 3))
            self._log_tau_sites = nn.Parameter(torch.zeros(N, self.num_lobes, device='cuda'))
            self._sites_mask = None

        if self.color_representation in _NSV_VARIANTS:
            N = self._xyz.shape[0]
            vectors = fibonacci_sphere(self.num_lobes)
            vectors = vectors.repeat(N, 1, 1).to('cuda')
            self._sites = nn.Parameter(vectors)
            if self.num_features == 3:
                pcd_colors = torch.tensor(
                    np.asarray(pcd.colors), dtype=torch.float, device='cuda'
                )
                if self.feature_activation == 'relu':
                    # relu(feat + 0.5) = rgb at init → feat = rgb - 0.5
                    feats = (pcd_colors - 0.5).unsqueeze(1).expand(N, self.num_lobes, 3).clone()
                else:
                    # sigmoid(feat) = rgb at init → feat = inverse_sigmoid(rgb)
                    feats = inverse_sigmoid(
                        pcd_colors.clamp(1e-6, 1 - 1e-6)
                    ).unsqueeze(1).expand(N, self.num_lobes, 3).clone()
            else:
                feats = torch.randn(N, self.num_lobes, self.num_features, device='cuda') * 0.01
            self._fsfeatures = nn.Parameter(feats)
            self._sites_mask = None
            if self.color_representation == 'nsv':
                self._log_tau = nn.Parameter(torch.tensor(_SOFTPLUS_INV_1, device='cuda'))
            if self.color_representation == 'wnsv_scalar':
                self._log_temps = nn.Parameter(
                    torch.full((N, self.num_lobes), _SOFTPLUS_INV_1, device='cuda')
                )
        
        if self.color_representation == 'sgs':
            mu = fibonacci_sphere(self.num_lobes) 
            mu = mu.repeat(self._xyz.shape[0], 1, 1).to('cuda') 
            self._mu = nn.Parameter(mu) 
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            self._alpha = nn.Parameter(colors.repeat(1, self.num_lobes).reshape(self._xyz.shape[0], -1, 3)) 
            self._dc = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), device='cuda', requires_grad=True))         
            self._sharpness = nn.Parameter(torch.full((self._xyz.shape[0], self.num_lobes, 1), 5.0, device='cuda', requires_grad=True))

    def prune(self, live_mask):
        self._xyz = self._xyz[live_mask]
        self._sh0 = self._sh0[live_mask]
        self._shN = self._shN[live_mask]
        self._sb_params = self._sb_params[live_mask]
        self._scaling = self._scaling[live_mask]
        self._rotation = self._rotation[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]
        if self.color_representation == 'voronoi':
            self._sites = self._sites[live_mask]
            self._colors = self._colors[live_mask]
            self._log_tau_sites = self._log_tau_sites[live_mask]

    def save(self, path):
        tensors = {
            "_xyz": self._xyz,
            "_opacity": self._opacity,
            "_beta": self._beta,
            "_scaling": self._scaling,
            "_rotation": self._rotation,
        }

        if self.color_representation in ["sh", "sb"]:
            tensors.update({
                "_sh0": self._sh0, "_shN": self._shN, "_sb_params": self._sb_params,
                "active_sh_degree": self.active_sh_degree, "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
            })
        elif self.color_representation == "voronoi":
            mask = self._sites_mask.bool()
            tensors.update({
                "_sh0": self._sh0, "_shN": self._shN, "_sb_params": self._sb_params,
                "active_sh_degree": self.active_sh_degree, "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
                "_sites_active": self._sites[mask], "_colors_active": self._colors[mask],
                "_log_tau_sites_active": self._log_tau_sites[mask],
                "_sites_mask": self._sites_mask,
                "num_lobes": getattr(self, "num_lobes", self._sites.shape[1]),
            })
        elif self.color_representation in _NSV_VARIANTS:
            mask = self._sites_mask.bool() if self._sites_mask is not None else torch.ones(
                self._sites.shape[0], self._sites.shape[1], dtype=torch.bool, device='cuda')
            extra = {}
            if self.color_representation == 'nsv':
                extra["_log_tau"] = self._log_tau.detach()
            if self.color_representation == 'wnsv_scalar':
                extra["_log_temps_active"] = self._log_temps[mask]
            tensors.update({
                "_sh0": self._sh0, "_shN": self._shN, "_sb_params": self._sb_params,
                "active_sh_degree": self.active_sh_degree, "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
                "_sites_active": self._sites[mask], "_features_active": self._features[mask],
                "_sites_mask": self._sites_mask,
                "num_lobes": self.num_lobes, "num_features": self.num_features,
                "mlp_state_dict": self.mlp.state_dict() if self.mlp is not None else None,
                "mlp_ema_state_dict": self.mlp_ema.state_dict() if self.mlp_ema is not None else None,
                **extra,
            })
        elif self.color_representation == "sgs":
            tensors.update({
                "_sh0": self._sh0, "_shN": self._shN, "_sb_params": self._sb_params,
                "active_sh_degree": self.active_sh_degree, "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
                "_mu": self._mu, "_alpha": self._alpha, "_sharpness": self._sharpness,
                "_dc": getattr(self, "_dc", torch.zeros((self._xyz.shape[0], 3), device="cuda")),
            })

        config = {
            "color_representation": self.color_representation,
            "spatial_lr_scale": getattr(self, "spatial_lr_scale", 1.0),
            "mlp_backend": getattr(self, "mlp_backend", "tcnn"),
            "forward_shading": self.forward_shading,
            "nsv_aggregate": getattr(self, "nsv_aggregate", "concat"),
            "nsv_top_k": getattr(self, "nsv_top_k", 0),
            "use_dir_enc": getattr(self, "use_dir_enc", False),
            "sh_degree_dir": getattr(self, "sh_degree_dir", 2),
        }

        opt_state = None
        if self.optimizer is not None:
            try:
                opt_state = self.optimizer.state_dict()
            except Exception:
                opt_state = None

        torch.save({"tensors": tensors, "config": config, "optimizer_state": opt_state}, path)


    def load(self, path, map_location="cuda"):
        data = torch.load(path, map_location=map_location)
        t = data["tensors"]
        c = data["config"]

        self._xyz = t["_xyz"]
        self._opacity = t["_opacity"]
        self._beta = t["_beta"]
        self._scaling = t["_scaling"]
        self._rotation = t["_rotation"]

        self.color_representation = c["color_representation"]
        self.spatial_lr_scale = c.get("spatial_lr_scale", 1.0)
        self.forward_shading = c.get("forward_shading", False)
        self.nsv_aggregate = c.get("nsv_aggregate", "concat")
        self.nsv_top_k = int(c.get("nsv_top_k", 0))
        self.use_dir_enc = c.get("use_dir_enc", False)
        self.sh_degree_dir = c.get("sh_degree_dir", 2)

        if self.color_representation in ["sh", "sb"]:
            self._sh0 = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]
        elif self.color_representation == "voronoi":
            self._sh0 = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]
            self._sites_mask = t["_sites_mask"]
            self.num_lobes = t.get("num_lobes", self._sites_mask.shape[1])
            N = self._sites_mask.shape[0]
            sites_full = torch.ones((N, self.num_lobes, 3), device=map_location)
            colors_full = torch.zeros_like(sites_full)
            log_tau_full = torch.zeros((N, self.num_lobes), device=map_location)
            mask = self._sites_mask.bool()
            sites_full[mask] = t["_sites_active"]
            colors_full[mask] = t["_colors_active"]
            if "_log_tau_sites_active" in t:
                log_tau_full[mask] = t["_log_tau_sites_active"]
            self._sites = nn.Parameter(sites_full)
            self._colors = nn.Parameter(colors_full)
            self._log_tau_sites = nn.Parameter(log_tau_full)
        elif self.color_representation in _NSV_VARIANTS:
            self._sh0 = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]
            self._sites_mask = t["_sites_mask"]
            self.num_lobes = t.get("num_lobes", self._sites_mask.shape[1])
            self.num_features = t["num_features"]
            N = self._sites_mask.shape[0]
            sites_full = torch.ones((N, self.num_lobes, 3), device=map_location)
            feats_full = torch.zeros((N, self.num_lobes, self.num_features), device=map_location)
            sites_full[self._sites_mask.bool()] = t["_sites_active"]
            feats_full[self._sites_mask.bool()] = t["_features_active"]
            self._sites = nn.Parameter(sites_full)
            self._features = nn.Parameter(feats_full)
            mlp_backend = c.get("mlp_backend", "tcnn")
            eff_k = self.nsv_top_k if 0 < self.nsv_top_k < self.num_lobes else self.num_lobes
            mlp_in = eff_k * self.num_features if self.nsv_aggregate == 'concat' else self.num_features
            mlp_sd = t.get("mlp_state_dict", None)
            if self.num_features == 3 or mlp_sd is None:
                self.mlp = None
                self.mlp_ema = None
            else:
                self.mlp = get_voronoi_mlp(
                    mlp_in, backend=mlp_backend,
                    use_dir_enc=self.use_dir_enc, sh_degree=self.sh_degree_dir,
                ).to(map_location)
                self.mlp.load_state_dict(mlp_sd)
                import copy
                ema_sd = t.get("mlp_ema_state_dict", None)
                if ema_sd is not None:
                    self.mlp_ema = copy.deepcopy(self.mlp)
                    self.mlp_ema.load_state_dict(ema_sd)
                    for p in self.mlp_ema.parameters():
                        p.requires_grad_(False)
                else:
                    self.mlp_ema = None
            if self.color_representation == 'nsv':
                self._log_tau = nn.Parameter(t["_log_tau"].to(map_location))
            if self.color_representation == 'wnsv_scalar':
                mask = self._sites_mask.bool()
                temps_full = torch.full((N, self.num_lobes), _SOFTPLUS_INV_1, device=map_location)
                temps_full[mask] = t["_log_temps_active"].to(map_location)
                self._log_temps = nn.Parameter(temps_full)
        elif self.color_representation == "sgs":
            self._mu = t["_mu"]; self._alpha = t["_alpha"]; self._sharpness = t["_sharpness"]
            self._dc = t.get("_dc", torch.zeros((self._xyz.shape[0], 3), device=map_location))
            self._sh0 = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]

        opt_state = data.get("optimizer_state", None)
        if opt_state is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                pass


    def training_setup(self, training_args, config=None):
        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._beta], "lr": training_args.beta_lr, "name": "beta"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]
        
        if self.color_representation == 'sh' or self.color_representation == 'sb':
            l += [
                {"params": [self._sh0], "lr": training_args.sh_lr, "name": "sh0"},
                {"params": [self._shN], "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr, "name": "sb_params"}
            ]
        
        if self.color_representation == 'voronoi':
            sites_lr = config.sites_lr if (config is not None and hasattr(config, 'sites_lr')) else training_args.sites_lr
            color_lr = config.color_lr if (config is not None and hasattr(config, 'color_lr')) else training_args.color_lr
            tau_lr = getattr(training_args, 'tau_lr', sites_lr)
            self.voronoi_sites_lr_init = sites_lr
            self.total_iterations = getattr(training_args, 'iterations', 45000)
            l += [
                {"params": [self._sh0], "lr": training_args.sh_lr, "name": "sh0"},
                {"params": [self._shN], "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr, "name": "sb_params"},
                {'params': [self._sites], 'lr': sites_lr, "name": "sites"},
                {'params': [self._colors], 'lr': color_lr, "name": "colors"},
                {'params': [self._log_tau_sites], 'lr': tau_lr, "name": "log_tau_sites"},
            ]

        elif self.color_representation in _NSV_VARIANTS:
            sites_lr = config.sites_lr if (config is not None and hasattr(config, 'sites_lr')) else training_args.sites_lr
            feat_lr  = config.color_lr  if (config is not None and hasattr(config, 'color_lr'))  else training_args.color_lr
            mlp_lr   = config.mlp_lr    if (config is not None and hasattr(config, 'mlp_lr'))    else getattr(training_args, 'mlp_lr', 1e-3)
            tau_lr   = getattr(training_args, 'tau_lr', sites_lr)
            l += [
                {"params": [self._sh0], "lr": training_args.sh_lr, "name": "sh0"},
                {"params": [self._shN], "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr, "name": "sb_params"},
                {'params': [self._sites], 'lr': sites_lr, "name": "sites"},
            ]
            self.feat_lr_init = feat_lr
            self.mlp_lr_init = mlp_lr
            self.total_iterations = getattr(training_args, 'iterations', 45000)
            self.feat_optimizer = torch.optim.Adam([self._features], lr=feat_lr, eps=1e-15)
            if self.mlp is not None:
                self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=mlp_lr, eps=1e-15)
                if getattr(self, 'use_mlp_ema', True):
                    import copy
                    self.mlp_ema = copy.deepcopy(self.mlp)
                    for p in self.mlp_ema.parameters():
                        p.requires_grad_(False)
                else:
                    self.mlp_ema = None
                self.mlp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.mlp_optimizer, T_max=self.total_iterations, eta_min=mlp_lr * 0.1,
                )
            else:
                self.mlp_optimizer = None
                self.mlp_ema = None
                self.mlp_scheduler = None
            if self.color_representation == 'nsv':
                self.tau_optimizer = torch.optim.Adam([self._log_tau], lr=tau_lr, eps=1e-15)
            elif self.color_representation == 'wnsv_scalar':
                self.tau_optimizer = torch.optim.Adam([self._log_temps], lr=tau_lr, eps=1e-15)

        elif self.color_representation == 'sgs':
            l += [
                {"params": [self._sh0], "lr": training_args.sh_lr, "name": "sh0"},
                {"params": [self._shN], "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr, "name": "sb_params"},
                {'params': [self._mu], 'lr': training_args.sites_lr, "name": "mu"},
                {'params': [self._alpha], 'lr': training_args.color_lr, "name": "alpha"},
                {'params': [self._sharpness], 'lr': 1e-2, "name": "sharpness"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.sites_scheduler_args = get_expon_lr_func(
            lr_init=5e-2, lr_final=1e-4,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=30_000
        )

    def update_learning_rate(self, iteration):
        xyz_lr = None
        _voronoi_frozen = (
            self.color_representation == 'voronoi'
            and hasattr(self, 'total_iterations')
            and iteration > self.total_iterations - 3000
        )
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                if not _voronoi_frozen:
                    lr = self.xyz_scheduler_args(iteration)
                    param_group["lr"] = lr
                xyz_lr = param_group["lr"]
        if self.color_representation == 'voronoi' and hasattr(self, 'voronoi_sites_lr_init'):
            if iteration <= self.total_iterations - 3000:
                T = self.total_iterations
                lr0 = self.voronoi_sites_lr_init
                cos_sites_lr = lr0 * 0.1 + 0.5 * lr0 * 0.9 * (1 + math.cos(math.pi * iteration / T))
                for pg in self.optimizer.param_groups:
                    if pg['name'] == 'sites':
                        pg['lr'] = cos_sites_lr
        # if self.color_representation in _NSV_VARIANTS and self.feat_optimizer is not None:
        #     T = self.total_iterations
        #     cos_feat_lr = (
        #         self.feat_lr_init * 0.1
        #         + 0.5 * self.feat_lr_init * 0.9 * (1 + math.cos(math.pi * iteration / T))
        #     )
        #     for pg in self.feat_optimizer.param_groups:
        #         pg['lr'] = cos_feat_lr
        return xyz_lr
            
    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._sh0.shape[1] * self._sh0.shape[2]):
            l.append(f"sh0_{i}")
        for i in range(self._shN.shape[1] * self._shN.shape[2]):
            l.append(f"shN_{i}")
        for i in range(self._sb_params.shape[1] * self._sb_params.shape[2]):
            l.append(f"sb_params_{i}")
        l.append("opacity")
        l.append("beta")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")

        if self.color_representation == "voronoi" and hasattr(self, "_sites") and self._sites.numel() > 0:
            L = self._sites.shape[1]
            for li in range(L):
                l += [f"site_{li}_x", f"site_{li}_y", f"site_{li}_z",
                      f"color_{li}_r", f"color_{li}_g", f"color_{li}_b"]
                if getattr(self, "_sites_mask", None) is not None:
                    l.append(f"site_{li}_mask")
                l.append(f"tau_{li}")

        if self.color_representation in _NSV_VARIANTS and hasattr(self, "_sites") and self._sites.numel() > 0:
            L = self._sites.shape[1]
            C = self._features.shape[2]
            for li in range(L):
                l += [f"site_{li}_x", f"site_{li}_y", f"site_{li}_z"]
                for ci in range(C):
                    l.append(f"feat_{li}_{ci}")
                if getattr(self, "_sites_mask", None) is not None:
                    l.append(f"site_{li}_mask")

        if self.color_representation == "sgs" and hasattr(self, "_mu") and self._mu.numel() > 0:
            L = self._mu.shape[1]
            l += ["dc_r", "dc_g", "dc_b"]
            for li in range(L):
                l += [f"mu_{li}_x", f"mu_{li}_y", f"mu_{li}_z",
                      f"alpha_{li}_r", f"alpha_{li}_g", f"alpha_{li}_b", f"sharp_{li}"]
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        sh0 = self._sh0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        shN = self._shN.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self._sb_params.numel() > 0:
            sb_params = self._sb_params.transpose(1, 2).detach().flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            sb_params = np.zeros((xyz.shape[0], 0), dtype=np.float32)
        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        extra_cols = []
        if self.color_representation == "voronoi" and hasattr(self, "_sites") and self._sites.numel() > 0:
            sites = self._sites.detach().cpu().numpy()
            colors = self._colors.detach().cpu().numpy()
            log_tau = self._log_tau_sites.detach().cpu().numpy()  # [N, L]
            N, L, _ = sites.shape
            for li in range(L):
                extra_cols.append(sites[:, li, :])
                extra_cols.append(colors[:, li, :])
                if getattr(self, "_sites_mask", None) is not None:
                    extra_cols.append(self._sites_mask[:, li].float().unsqueeze(-1).detach().cpu().numpy())
                extra_cols.append(log_tau[:, li: li + 1])

        if self.color_representation in _NSV_VARIANTS and hasattr(self, "_sites") and self._sites.numel() > 0:
            sites = self._sites.detach().cpu().numpy()
            feats = self._features.detach().cpu().numpy()
            N, L, C = feats.shape
            for li in range(L):
                extra_cols.append(sites[:, li, :])
                extra_cols.append(feats[:, li, :])
                if getattr(self, "_sites_mask", None) is not None:
                    extra_cols.append(self._sites_mask[:, li].float().unsqueeze(-1).detach().cpu().numpy())

        if self.color_representation == "sgs" and hasattr(self, "_mu") and self._mu.numel() > 0:
            mu = self._mu.detach().cpu().numpy()
            alpha = self._alpha.detach().cpu().numpy()
            sharp = self._sharpness.detach().cpu().numpy()
            dc = self._dc.detach().cpu().numpy()
            N, L, _ = mu.shape
            extra_cols.append(dc)
            for li in range(L):
                extra_cols.append(mu[:, li, :])
                extra_cols.append(alpha[:, li, :])
                extra_cols.append(sharp[:, li, :])

        base_attrs = [xyz, normals, sh0, shN, sb_params, opacities, betas, scale, rotation]
        if len(extra_cols) > 0:
            attributes = np.concatenate(base_attrs + extra_cols, axis=1)
        else:
            attributes = np.concatenate(base_attrs, axis=1)

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        assert attributes.shape[1] == len(dtype_full), f"Mismatch: attributes {attributes.shape[1]} vs dtype {len(dtype_full)}"
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        elem = plydata.elements[0]
        xyz = np.stack((np.asarray(elem["x"]), np.asarray(elem["y"]), np.asarray(elem["z"])), axis=1)
        opacities = np.asarray(elem["opacity"])[..., np.newaxis]
        betas = np.asarray(elem["beta"])[..., np.newaxis]
        sh0 = np.zeros((xyz.shape[0], 3, 1))
        sh0[:, 0, 0] = np.asarray(elem["sh0_0"])
        sh0[:, 1, 0] = np.asarray(elem["sh0_1"])
        sh0[:, 2, 0] = np.asarray(elem["sh0_2"])

        prop_names = [p.name for p in elem.properties]
        shN_names = sorted([n for n in prop_names if n.startswith("shN_")], key=lambda x: int(x.split("_")[-1]))
        assert len(shN_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        shs_extra = np.zeros((xyz.shape[0], len(shN_names)))
        for idx, attr_name in enumerate(shN_names):
            shs_extra[:, idx] = np.asarray(elem[attr_name])
        shs_extra = shs_extra.reshape((shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        sb_names = sorted([n for n in prop_names if n.startswith("sb_params_")], key=lambda x: int(x.split("_")[-1]))
        if self.sb_number > 0:
            assert len(sb_names) == self.sb_number * 6
            sb_params = np.zeros((xyz.shape[0], len(sb_names)))
            for idx, attr_name in enumerate(sb_names):
                sb_params[:, idx] = np.asarray(elem[attr_name])
            sb_params = sb_params.reshape((sb_params.shape[0], 6, self.sb_number))
        else:
            sb_params = np.zeros((xyz.shape[0], 6, 0))

        scale_names = sorted([n for n in prop_names if n.startswith("scale_")], key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(elem[attr_name])

        rot_names = sorted([n for n in prop_names if n.startswith("rot_")], key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(elem[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sh0 = nn.Parameter(torch.tensor(sh0, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._shN = nn.Parameter(torch.tensor(shs_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._sb_params = nn.Parameter(torch.tensor(sb_params, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._beta = nn.Parameter(torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        if self.color_representation == "voronoi":
            site_xyz_names = [n for n in prop_names if n.startswith("site_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))]
            color_names = [n for n in prop_names if n.startswith("color_") and (n.endswith("_r") or n.endswith("_g") or n.endswith("_b"))]
            mask_names = [n for n in prop_names if n.startswith("site_") and n.endswith("_mask")]
            tau_names = sorted([n for n in prop_names if n.startswith("tau_")], key=lambda x: int(x.split("_")[1]))
            if len(site_xyz_names) > 0:
                def site_key(name):
                    parts = name.split("_"); li = int(parts[1]); return li * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]
                def color_key(name):
                    parts = name.split("_"); li = int(parts[1]); return li * 3 + {"r": 0, "g": 1, "b": 2}[parts[2]]
                site_xyz_names = sorted(site_xyz_names, key=site_key)
                color_names = sorted(color_names, key=color_key)
                sites_flat = np.zeros((xyz.shape[0], len(site_xyz_names)))
                for idx, attr_name in enumerate(site_xyz_names):
                    sites_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(site_xyz_names) // 3
                sites = sites_flat.reshape(xyz.shape[0], L, 3)
                colors_flat = np.zeros((xyz.shape[0], len(color_names)))
                for idx, attr_name in enumerate(color_names):
                    colors_flat[:, idx] = np.asarray(elem[attr_name])
                colors = colors_flat.reshape(xyz.shape[0], L, 3)
                sites_mask = None
                if len(mask_names) > 0:
                    mask_names = sorted(mask_names, key=lambda x: int(x.split("_")[1]))
                    mask_flat = np.zeros((xyz.shape[0], len(mask_names)))
                    for idx, attr_name in enumerate(mask_names):
                        mask_flat[:, idx] = np.asarray(elem[attr_name])
                    sites_mask = mask_flat.astype(np.float32).reshape(xyz.shape[0], L)
                if len(tau_names) > 0:
                    tau_flat = np.zeros((xyz.shape[0], len(tau_names)))
                    for idx, attr_name in enumerate(tau_names):
                        tau_flat[:, idx] = np.asarray(elem[attr_name])
                else:
                    tau_flat = np.zeros((xyz.shape[0], L), dtype=np.float32)
                self._sites = nn.Parameter(torch.tensor(sites, dtype=torch.float, device="cuda").requires_grad_(True))
                self._colors = nn.Parameter(torch.tensor(colors, dtype=torch.float, device="cuda").requires_grad_(True))
                self._sites_mask = torch.tensor(sites_mask > 0.5, dtype=torch.bool, device="cuda") if sites_mask is not None else None
                self._log_tau_sites = nn.Parameter(torch.tensor(tau_flat, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.color_representation == "nsv":
            site_xyz_names = [n for n in prop_names if n.startswith("site_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))]
            feat_names = [n for n in prop_names if n.startswith("feat_")]
            mask_names = [n for n in prop_names if n.startswith("site_") and n.endswith("_mask")]
            if len(site_xyz_names) > 0:
                def site_key(name):
                    parts = name.split("_"); li = int(parts[1]); return li * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]
                def feat_key(name):
                    parts = name.split("_"); return int(parts[1]) * 10000 + int(parts[2])
                site_xyz_names = sorted(site_xyz_names, key=site_key)
                feat_nfames = sorted(feat_names, key=feat_key)
                sites_flat = np.zeros((xyz.shape[0], len(site_xyz_names)))
                for idx, attr_name in enumerate(site_xyz_names):
                    sites_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(site_xyz_names) // 3
                sites = sites_flat.reshape(xyz.shape[0], L, 3)
                C = len(feat_names) // L
                feats_flat = np.zeros((xyz.shape[0], len(feat_names)))
                for idx, attr_name in enumerate(feat_names):
                    feats_flat[:, idx] = np.asarray(elem[attr_name])
                feats = feats_flat.reshape(xyz.shape[0], L, C)
                sites_mask = None
                if len(mask_names) > 0:
                    mask_names = sorted(mask_names, key=lambda x: int(x.split("_")[1]))
                    mask_flat = np.zeros((xyz.shape[0], len(mask_names)))
                    for idx, attr_name in enumerate(mask_names):
                        mask_flat[:, idx] = np.asarray(elem[attr_name])
                    sites_mask = mask_flat.astype(np.float32).reshape(xyz.shape[0], L)
                self._sites = nn.Parameter(torch.tensor(sites, dtype=torch.float, device="cuda").requires_grad_(True))
                self._features = nn.Parameter(torch.tensor(feats, dtype=torch.float, device="cuda").requires_grad_(True))
                self._sites_mask = torch.tensor(sites_mask > 0.5, dtype=torch.bool, device="cuda") if sites_mask is not None else None
                self.num_lobes = L
                self.num_features = C
                top_k = getattr(self, 'nsv_top_k', 0)
                eff_k = top_k if 0 < top_k < L else L
                mlp_in = eff_k * C if getattr(self, 'nsv_aggregate', 'concat') == 'concat' else C
                self.mlp = None if C == 3 else get_voronoi_mlp(mlp_in, backend=self.mlp_backend).cuda()

        if self.color_representation == "sgs":
            dc = None
            if all(n in prop_names for n in ["dc_r", "dc_g", "dc_b"]):
                dc = np.stack((np.asarray(elem["dc_r"]), np.asarray(elem["dc_g"]), np.asarray(elem["dc_b"])), axis=1)
            mu_names = [n for n in prop_names if n.startswith("mu_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))]
            alpha_names = [n for n in prop_names if n.startswith("alpha_") and (n.endswith("_r") or n.endswith("_g") or n.endswith("_b"))]
            sharp_names = [n for n in prop_names if n.startswith("sharp_")]
            if len(mu_names) > 0:
                def mu_key(name):
                    parts = name.split("_"); li = int(parts[1]); return li * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]
                def alpha_key(name):
                    parts = name.split("_"); li = int(parts[1]); return li * 3 + {"r": 0, "g": 1, "b": 2}[parts[2]]
                mu_names = sorted(mu_names, key=mu_key)
                alpha_names = sorted(alpha_names, key=alpha_key)
                sharp_names = sorted(sharp_names, key=lambda x: int(x.split("_")[1]))
                mu_flat = np.zeros((xyz.shape[0], len(mu_names)))
                for idx, attr_name in enumerate(mu_names): mu_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(mu_names) // 3; mu = mu_flat.reshape(xyz.shape[0], L, 3)
                alpha_flat = np.zeros((xyz.shape[0], len(alpha_names)))
                for idx, attr_name in enumerate(alpha_names): alpha_flat[:, idx] = np.asarray(elem[attr_name])
                alpha = alpha_flat.reshape(xyz.shape[0], L, 3)
                sharp_flat = np.zeros((xyz.shape[0], len(sharp_names)))
                for idx, attr_name in enumerate(sharp_names): sharp_flat[:, idx] = np.asarray(elem[attr_name])
                sharp = sharp_flat.reshape(xyz.shape[0], L, 1)
                self._mu = nn.Parameter(torch.tensor(mu, dtype=torch.float, device="cuda").requires_grad_(True))
                self._alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float, device="cuda").requires_grad_(True))
                self._sharpness = nn.Parameter(torch.tensor(sharp, dtype=torch.float, device="cuda").requires_grad_(True))
                if dc is not None:
                    self._dc = nn.Parameter(torch.tensor(dc, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, params, reset_params=True):
        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._xyz = optimizable_tensors["xyz"]
        self._sh0 = optimizable_tensors["sh0"]
        self._shN = optimizable_tensors["shN"]
        self._sb_params = optimizable_tensors["sb_params"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.color_representation == 'voronoi':
            self._sites = optimizable_tensors['sites']
            self._colors = optimizable_tensors['colors']
            self._log_tau_sites = optimizable_tensors['log_tau_sites']
        elif self.color_representation in _NSV_VARIANTS:
            self._sites = optimizable_tensors['sites']
            if self.feat_optimizer is not None and 'features' in params:
                new_features = params['features']
                feat_group = self.feat_optimizer.param_groups[0]
                stored_state = self.feat_optimizer.state.get(feat_group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(new_features)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(new_features)), dim=0)
                    del self.feat_optimizer.state[feat_group["params"][0]]
                    feat_group["params"][0] = nn.Parameter(torch.cat((feat_group["params"][0], new_features), dim=0).requires_grad_(True))
                    self.feat_optimizer.state[feat_group["params"][0]] = stored_state
                else:
                    feat_group["params"][0] = nn.Parameter(torch.cat((feat_group["params"][0], new_features), dim=0).requires_grad_(True))
                self._features = feat_group["params"][0]
            if self.color_representation == 'wnsv_scalar' and self.tau_optimizer is not None and 'log_temps' in params:
                new_temps = params['log_temps']
                tau_group = self.tau_optimizer.param_groups[0]
                stored_state = self.tau_optimizer.state.get(tau_group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(new_temps)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(new_temps)), dim=0)
                    del self.tau_optimizer.state[tau_group["params"][0]]
                    tau_group["params"][0] = nn.Parameter(torch.cat((tau_group["params"][0], new_temps), dim=0).requires_grad_(True))
                    self.tau_optimizer.state[tau_group["params"][0]] = stored_state
                else:
                    tau_group["params"][0] = nn.Parameter(torch.cat((tau_group["params"][0], new_temps), dim=0).requires_grad_(True))
                self._log_temps = tau_group["params"][0]
        if self.color_representation == 'sgs':
            self._mu = optimizable_tensors['mu']
            self._alpha = optimizable_tensors['alpha']
            self._sharpness = optimizable_tensors['sharpness']

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz, "sh0": self._sh0, "shN": self._shN, "sb_params": self._sb_params,
            "opacity": self._opacity, "beta": self._beta, "scaling": self._scaling, "rotation": self._rotation,
        }
        if self.color_representation == 'voronoi':
            tensors_dict.update({'sites': self._sites, 'colors': self._colors, 'log_tau_sites': self._log_tau_sites})
        elif self.color_representation in _NSV_VARIANTS:
            tensors_dict.update({'sites': self._sites})
        if self.color_representation == 'sgs':
            tensors_dict.update({'mu': self._mu, 'alpha': self._alpha, 'sharpness': self._sharpness})

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._sh0 = optimizable_tensors["sh0"]
        self._shN = optimizable_tensors["shN"]
        self._sb_params = optimizable_tensors["sb_params"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.color_representation == 'voronoi':
            self._sites         = optimizable_tensors['sites']
            self._colors        = optimizable_tensors['colors']
            self._log_tau_sites = optimizable_tensors['log_tau_sites']
        elif self.color_representation in _NSV_VARIANTS:
            self._sites = optimizable_tensors['sites']
            if self.feat_optimizer is not None:
                feat_group = self.feat_optimizer.param_groups[0]
                stored_state = self.feat_optimizer.state.get(feat_group["params"][0], None)
                if stored_state is not None:
                    if inds is not None:
                        stored_state["exp_avg"][inds] = 0
                        stored_state["exp_avg_sq"][inds] = 0
                    else:
                        stored_state["exp_avg"] = torch.zeros_like(self._features)
                        stored_state["exp_avg_sq"] = torch.zeros_like(self._features)
                    del self.feat_optimizer.state[feat_group["params"][0]]
                    feat_group["params"][0] = nn.Parameter(self._features.requires_grad_(True))
                    self.feat_optimizer.state[feat_group["params"][0]] = stored_state
                self._features = feat_group["params"][0]
            if self.color_representation == 'wnsv_scalar' and self.tau_optimizer is not None:
                tau_group = self.tau_optimizer.param_groups[0]
                stored_state = self.tau_optimizer.state.get(tau_group["params"][0], None)
                if stored_state is not None:
                    if inds is not None:
                        stored_state["exp_avg"][inds] = 0
                        stored_state["exp_avg_sq"][inds] = 0
                    else:
                        stored_state["exp_avg"] = torch.zeros_like(self._log_temps)
                        stored_state["exp_avg_sq"] = torch.zeros_like(self._log_temps)
                    del self.tau_optimizer.state[tau_group["params"][0]]
                    tau_group["params"][0] = nn.Parameter(self._log_temps.requires_grad_(True))
                    self.tau_optimizer.state[tau_group["params"][0]] = stored_state
                self._log_temps = tau_group["params"][0]
        if self.color_representation == 'sgs':
            self._mu = optimizable_tensors['mu']
            self._alpha = optimizable_tensors['alpha']
            self._sharpness = optimizable_tensors['sharpness']
        torch.cuda.empty_cache()
        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1))
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        ret = [self._xyz[idxs], self._sh0[idxs], self._shN[idxs], self._sb_params[idxs],
               new_opacity, self._beta[idxs], self._scaling[idxs], self._rotation[idxs]]
        if self.color_representation == 'voronoi':
            ret += [self._sites[idxs], self._colors[idxs], self._log_tau_sites[idxs]]
        elif self.color_representation in _NSV_VARIANTS:
            ret += [self._sites[idxs], self._features[idxs]]
            if self.color_representation == 'wnsv_scalar':
                ret += [self._log_temps[idxs]]
        if self.color_representation == 'sgs':
            ret += [self._mu[idxs], self._alpha[idxs], self._sharpness[idxs]]
        return (*ret,)

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (self._xyz[dead_indices], self._sh0[dead_indices], self._shN[dead_indices],
         self._sb_params[dead_indices], self._opacity[dead_indices], self._beta[dead_indices],
         self._scaling[dead_indices], self._rotation[dead_indices], *rest) = self._update_params(reinit_idx, ratio=ratio)
        
        if self.color_representation == 'voronoi':
            self._sites[dead_indices] = rest[0]; self._colors[dead_indices] = rest[1]; self._log_tau_sites[dead_indices] = rest[2]
        elif self.color_representation in _NSV_VARIANTS:
            self._sites[dead_indices] = rest[0]; self._features[dead_indices] = rest[1]
            if self.color_representation == 'wnsv_scalar':
                self._log_temps[dead_indices] = rest[2]
        if self.color_representation == 'sgs':
            self._mu[dead_indices] = rest[0]; self._alpha[dead_indices] = rest[1]; self._sharpness[dead_indices] = rest[2]
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def subsample_to(self, cap_max):
        """Randomly subsample Gaussians to cap_max, updating all params and optimizers."""
        N = self._xyz.shape[0]
        if N <= cap_max:
            return
        perm = torch.randperm(N, device='cuda')[:cap_max]
        mask = torch.zeros(N, dtype=torch.bool, device='cuda')
        mask[perm] = True

        optimizable_tensors = self._prune_optimizer(mask)
        self._xyz       = optimizable_tensors["xyz"]
        self._sh0       = optimizable_tensors["sh0"]
        self._shN       = optimizable_tensors["shN"]
        self._sb_params = optimizable_tensors["sb_params"]
        self._opacity   = optimizable_tensors["opacity"]
        self._beta      = optimizable_tensors["beta"]
        self._scaling   = optimizable_tensors["scaling"]
        self._rotation  = optimizable_tensors["rotation"]

        if self.color_representation == 'voronoi':
            self._sites          = optimizable_tensors['sites']
            self._colors         = optimizable_tensors['colors']
            self._log_tau_sites  = optimizable_tensors['log_tau_sites']
        elif self.color_representation in _NSV_VARIANTS:
            self._sites = optimizable_tensors['sites']
            if self.feat_optimizer is not None:
                feat_group = self.feat_optimizer.param_groups[0]
                stored = self.feat_optimizer.state.get(feat_group["params"][0], None)
                if stored is not None:
                    stored["exp_avg"]    = stored["exp_avg"][mask]
                    stored["exp_avg_sq"] = stored["exp_avg_sq"][mask]
                    del self.feat_optimizer.state[feat_group["params"][0]]
                    feat_group["params"][0] = nn.Parameter(feat_group["params"][0][mask].requires_grad_(True))
                    self.feat_optimizer.state[feat_group["params"][0]] = stored
                else:
                    feat_group["params"][0] = nn.Parameter(feat_group["params"][0][mask].requires_grad_(True))
                self._features = feat_group["params"][0]
            else:
                self._features = nn.Parameter(self._features[mask].requires_grad_(True))
            if self.color_representation == 'wnsv_scalar':
                if self.tau_optimizer is not None:
                    tau_group = self.tau_optimizer.param_groups[0]
                    stored = self.tau_optimizer.state.get(tau_group["params"][0], None)
                    if stored is not None:
                        stored["exp_avg"]    = stored["exp_avg"][mask]
                        stored["exp_avg_sq"] = stored["exp_avg_sq"][mask]
                        del self.tau_optimizer.state[tau_group["params"][0]]
                        tau_group["params"][0] = nn.Parameter(tau_group["params"][0][mask].requires_grad_(True))
                        self.tau_optimizer.state[tau_group["params"][0]] = stored
                    else:
                        tau_group["params"][0] = nn.Parameter(tau_group["params"][0][mask].requires_grad_(True))
                    self._log_temps = tau_group["params"][0]
                else:
                    self._log_temps = nn.Parameter(self._log_temps[mask].requires_grad_(True))
        elif self.color_representation == 'sgs':
            self._mu        = optimizable_tensors['mu']
            self._alpha     = optimizable_tensors['alpha']
            self._sharpness = optimizable_tensors['sharpness']

        if self._sites_mask is not None:
            self._sites_mask = self._sites_mask[mask]

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (new_xyz, new_sh0, new_shN, new_sb_params, new_opacity, new_beta,
         new_scaling, new_rotation, *rest) = self._update_params(add_idx, ratio=ratio)
        
        params = {'xyz': new_xyz, 'opacity': new_opacity, 'scaling': new_scaling, 'rotation': new_rotation,
                  "sh0": new_sh0, "shN": new_shN, "sb_params": new_sb_params, "beta": new_beta}
        if self.color_representation == 'voronoi':
            params.update({'sites': rest[0], 'colors': rest[1], 'log_tau_sites': rest[2]})
        elif self.color_representation in _NSV_VARIANTS:
            params.update({'sites': rest[0], 'features': rest[1]})
            if self.color_representation == 'wnsv_scalar':
                params.update({'log_temps': rest[2]})
        elif self.color_representation == 'sgs':
            params.update({'mu': rest[0], 'alpha': rest[1], 'sharpness': rest[2]})

        self._opacity[add_idx] = new_opacity
        self.densification_postfix(params, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    def render(self, viewpoint_camera, render_mode="RGB", measure_fps=None):
        # measure_fps: True/False to force; None = auto (= not self.training).
        if measure_fps is None:
            measure_fps = not self.training

        # Intrinsics build is Python + tiny scalar kernels — keep it OUT of the
        # timed region so the FPS reflects only the GPU rendering work
        # (per-Gaussian color computation + rasterization), matching the
        # original 3DGS render() timing pattern.
        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)
        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)
        K[0, 0] = fx; K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        use_deferred = (self.color_representation in _NSV_VARIANTS and not self.forward_shading)

        # Start timer right before the heavy GPU work.
        if measure_fps:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

        if self.color_representation in ['voronoi', 'sgs'] + list(_NSV_VARIANTS):
            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

            # if self.color_representation in _NSV_VARIANTS:
            #     if self.forward_shading:
            #         fast_rgb = (
            #             not torch.is_grad_enabled()
            #             and self.color_representation == 'wnsv_norm'
            #             and self.nsv_aggregate == 'sum'
            #             and self.num_features == 3
            #             and self.mlp is None
            #             and getattr(self, 'nsv_top_k', 0) <= 0
            #         )
            #         # if fast_rgb:
            #         #     use_relu = (self.feature_activation == 'relu')
            #         #     if HAS_NSV_CUDA:
            #         #         if use_relu:
            #         #             # relu(weighted_feat + 0.5): pass features+0.5 (linearity of weighted sum)
            #         #             colors_precomp = nsv_sum_rgb_relu_cuda(
            #         #                 self._sites, self._features + 0.5, dir_pp_normalized)
            #         #         else:
            #         #             colors_precomp = nsv_sum_rgb_sigmoid_cuda(
            #         #                 self._sites, self._features, dir_pp_normalized)
            #         #     else:
            #         #         if use_relu:
            #         #             colors_precomp = nsv_sum_rgb_relu_torch(
            #         #                 self._sites, self._features + 0.5, dir_pp_normalized)
            #         #         else:
            #         #             colors_precomp = nsv_sum_rgb_sigmoid_torch(
            #         #                 self._sites, self._features, dir_pp_normalized)
            #         # else:
            #         feat_precomp = self.compute_radiance(dir_pp_normalized)
            #         if self.mlp is not None:
            #             _mlp = self.mlp if (self.training or self.mlp_ema is None) else self.mlp_ema
            #             dirs_in = dir_pp_normalized if getattr(self.mlp, 'use_dir_enc', False) else None
            #             colors_precomp = _mlp(feat_precomp, dirs_in).float()
            #         else:
            #             if self.feature_activation == 'relu':
            #                 colors_precomp = torch.relu(feat_precomp + 0.5).float()
            #             else:
            #                 colors_precomp = torch.sigmoid(feat_precomp).float()
            #     else:
            #         feat_precomp = self.compute_radiance(dir_pp_normalized)
            # elif 
            # self.color_representation == 'voronoi':
            colors_precomp = self.compute_radiance(dir_pp_normalized)
            # else:
            #     colors_precomp = self.compute_radiance(dir_pp_normalized)
            active_degree = None
        else:
            colors_precomp = self.get_shs
            active_degree = self.active_sh_degree

        if use_deferred:
            top_k = getattr(self, 'nsv_top_k', 0)
            eff_k = top_k if 0 < top_k < self.num_lobes else self.num_lobes
            kc = self.num_features if self.nsv_aggregate == 'sum' else eff_k * self.num_features
            feat_bg = torch.zeros(kc, device='cuda')

            feat_map, alphas, meta = rasterization(
                means=self.get_xyz, quats=self.get_rotation,
                scales=self.get_scaling, opacities=self.get_opacity.squeeze(),
                betas=self.get_beta.squeeze(), colors=feat_precomp,
                viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
                Ks=K.unsqueeze(0), width=viewpoint_camera.image_width,
                height=viewpoint_camera.image_height, backgrounds=feat_bg.unsqueeze(0),
                render_mode=render_mode, covars=None, sh_degree=active_degree,
                sb_number=self.sb_number, sb_params=self.get_sb_params, packed=False,
            )

            H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
            feat_flat = feat_map[0].reshape(-1, kc)
            if self.mlp is not None:
                _mlp = self.mlp if (self.training or self.mlp_ema is None) else self.mlp_ema
                if getattr(self.mlp, 'use_dir_enc', False):
                    # per-pixel ray directions in world space
                    ys, xs = torch.meshgrid(
                        torch.arange(H, device='cuda', dtype=torch.float32) + 0.5,
                        torch.arange(W, device='cuda', dtype=torch.float32) + 0.5,
                        indexing='ij',
                    )
                    d_cam = torch.stack(
                        [(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1],
                         torch.ones_like(xs)], dim=-1
                    ).reshape(-1, 3)
                    # world_view_transform[:3,:3] = R_cam2world; row-vec: d_world = d_cam @ R_c2w.T
                    wvt = viewpoint_camera.world_view_transform
                    pixel_dirs = F.normalize(d_cam @ wvt[:3, :3].T, dim=-1)
                    rgb_flat = _mlp(feat_flat, pixel_dirs).float()
                else:
                    rgb_flat = _mlp(feat_flat).float()
            else:
                if self.feature_activation == 'relu':
                    rgb_flat = torch.relu(feat_flat + 0.5).float()
                else:
                    rgb_flat = torch.sigmoid(feat_flat).float()
            rgb = rgb_flat.reshape(H, W, 3)
            alpha = alphas[0]
            bg = self.background.view(1, 1, 3)
            rgb = rgb * alpha + bg * (1.0 - alpha)
            rgb = rgb.permute(2, 0, 1).contiguous()
        else:
            rgbs, alphas, meta = rasterization(
                means=self.get_xyz, quats=self.get_rotation,
                scales=self.get_scaling, opacities=self.get_opacity.squeeze(),
                betas=self.get_beta.squeeze(), colors=colors_precomp,
                viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
                Ks=K.unsqueeze(0), width=viewpoint_camera.image_width,
                height=viewpoint_camera.image_height, backgrounds=self.background.unsqueeze(0),
                render_mode=render_mode, covars=None, sh_degree=active_degree,
                sb_number=self.sb_number, sb_params=self.get_sb_params, packed=False,
            )
            rgb = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        if measure_fps:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed    = elapsed_ms / 1000.0  # seconds
            fps        = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        else:
            elapsed = 0.0
            fps     = 0.0

        return {
            "render": rgb,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0,
            "elapsed": elapsed,
            "FPS":     fps,
        }
