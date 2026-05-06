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

# raw init so that F.softplus(x) == 1.0
_SOFTPLUS_INV_1 = float(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))


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

    def __init__(self, color_representation='sh', num_lobes=8):
        """
        Args:
            color_representation: 'voronoi' | 'sh' | 'sb' | 'sgs'
            num_lobes: Voronoi / SGS lobes (ignored for sh / sb)
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
        self.spatial_lr_scale = 0
        self.color_representation = color_representation

        if color_representation == 'voronoi':
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self._log_tau_sites = torch.empty(0)
            self.num_lobes = num_lobes

        if color_representation == 'sgs':
            self._mu = torch.empty(0)
            self._alpha = torch.empty(0)
            self._lambda = torch.empty(0)
            self._dc = torch.empty(0)
            self.num_lobes = num_lobes

        self.setup_functions()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    def get_mu(self):
        return F.normalize(self._mu, dim=-1)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self):
        self.training = False
        if self.color_representation == 'voronoi':
            self._nn_sites = F.normalize(self._sites, dim=-1)
            self._nn_tau   = torch.exp(self._log_tau_sites).detach()

    def train(self):
        self.training = True

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------
    # Color computation
    # ------------------------------------------------------------------

    def compute_voronoi_color(self, directions: torch.Tensor) -> torch.Tensor:
        if self.training:
            sites_n = F.normalize(self._sites, dim=-1)
            tau     = torch.exp(self._log_tau_sites)
        else:
            sites_n = self._nn_sites
            tau     = self._nn_tau
        return spherical_voronoi(directions.reshape(-1, 3), sites_n, tau, self._colors)

    def compute_radiance(self, directions):
        if self.color_representation == 'voronoi':
            return self.compute_voronoi_color(directions.reshape(-1, 3))

        if self.color_representation == 'sgs':
            directions = directions.reshape(-1, 1, 3)
            mu = self.get_mu
            dot_product = (mu * directions).sum(dim=-1).unsqueeze(-1)
            sharpness = F.softplus(self._sharpness)
            V = self._alpha * torch.exp(sharpness * (dot_product - 1))
            V = V.sum(dim=1)
            return torch.clamp_min(V, 0.0)

    # ------------------------------------------------------------------
    # Initialisation from point cloud
    # ------------------------------------------------------------------

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
            vectors = fibonacci_sphere(self.num_lobes).repeat(N, 1, 1).to('cuda')
            self._sites = nn.Parameter(vectors)
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda() - 0.5
            self._colors = nn.Parameter(colors.repeat(1, self.num_lobes).reshape(N, -1, 3))
            self._log_tau_sites = nn.Parameter(torch.zeros(N, self.num_lobes, device='cuda'))

        if self.color_representation == 'sgs':
            N = self._xyz.shape[0]
            mu = fibonacci_sphere(self.num_lobes).repeat(N, 1, 1).to('cuda')
            self._mu = nn.Parameter(mu)
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            self._alpha = nn.Parameter(
                colors.repeat(1, self.num_lobes).reshape(N, -1, 3)
            )
            self._dc = nn.Parameter(
                torch.zeros((N, 3), device='cuda', requires_grad=True)
            )
            self._sharpness = nn.Parameter(
                torch.full((N, self.num_lobes, 1), 5.0, device='cuda', requires_grad=True)
            )

    # ------------------------------------------------------------------
    # Checkpoint: save / load / restore
    # ------------------------------------------------------------------

    def save(self, path):
        tensors = {
            "_xyz": self._xyz.detach(),
            "_opacity": self._opacity.detach(),
            "_beta": self._beta.detach(),
            "_scaling": self._scaling.detach(),
            "_rotation": self._rotation.detach(),
        }

        if self.color_representation in ("sh", "sb"):
            tensors.update({
                "_sh0": self._sh0.detach(),
                "_shN": self._shN.detach(),
                "_sb_params": self._sb_params.detach(),
                "active_sh_degree": self.active_sh_degree,
                "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
            })

        elif self.color_representation == "voronoi":
            tensors.update({
                "_sh0": self._sh0.detach(),
                "_shN": self._shN.detach(),
                "_sb_params": self._sb_params.detach(),
                "active_sh_degree": self.active_sh_degree,
                "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
                "_sites": self._sites.detach(),
                "_colors": self._colors.detach(),
                "_log_tau_sites": self._log_tau_sites.detach(),
                "num_lobes": self.num_lobes,
            })

        elif self.color_representation == "sgs":
            tensors.update({
                "_sh0": self._sh0.detach(),
                "_shN": self._shN.detach(),
                "_sb_params": self._sb_params.detach(),
                "active_sh_degree": self.active_sh_degree,
                "sb_number": self.sb_number,
                "max_sh_degree": self.max_sh_degree,
                "_mu": self._mu.detach(),
                "_alpha": self._alpha.detach(),
                "_sharpness": self._sharpness.detach(),
                "_dc": getattr(self, "_dc",
                               torch.zeros((self._xyz.shape[0], 3), device="cuda")).detach(),
                "num_lobes": self.num_lobes,
            })

        config = {
            "color_representation": self.color_representation,
            "spatial_lr_scale": getattr(self, "spatial_lr_scale", 1.0),
        }

        opt_state = None
        if self.optimizer is not None:
            try:
                opt_state = self.optimizer.state_dict()
            except Exception:
                opt_state = None

        torch.save({"tensors": tensors, "config": config, "optimizer_state": opt_state}, path)

    def restore(self, path, training_args):
        """Load a checkpoint saved by save() and rebuild the optimizer."""
        data = torch.load(path, map_location="cuda")
        t = data["tensors"]
        c = data["config"]

        self.color_representation = c["color_representation"]
        self.spatial_lr_scale = c.get("spatial_lr_scale", 1.0)

        def _p(tensor):
            return nn.Parameter(tensor.cuda().requires_grad_(True))

        self._xyz         = _p(t["_xyz"])
        self._opacity     = _p(t["_opacity"])
        self._beta        = _p(t["_beta"])
        self._scaling     = _p(t["_scaling"])
        self._rotation    = _p(t["_rotation"])

        if self.color_representation in ("sh", "sb"):
            self._sh0        = _p(t["_sh0"])
            self._shN        = _p(t["_shN"])
            self._sb_params  = _p(t["_sb_params"])
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number   = t["sb_number"]
            self.max_sh_degree = t["max_sh_degree"]

        elif self.color_representation == "voronoi":
            self._sh0        = _p(t["_sh0"])
            self._shN        = _p(t["_shN"])
            self._sb_params  = _p(t["_sb_params"])
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number   = t["sb_number"]
            self.max_sh_degree = t["max_sh_degree"]
            self.num_lobes   = t.get("num_lobes", 8)
            self._sites         = _p(t["_sites"].cuda())
            self._colors        = _p(t["_colors"].cuda())
            self._log_tau_sites = _p(t["_log_tau_sites"].cuda())

        elif self.color_representation == "sgs":
            self._sh0        = _p(t["_sh0"])
            self._shN        = _p(t["_shN"])
            self._sb_params  = _p(t["_sb_params"])
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number   = t["sb_number"]
            self.max_sh_degree = t["max_sh_degree"]
            self._mu         = _p(t["_mu"])
            self._alpha      = _p(t["_alpha"])
            self._sharpness  = _p(t["_sharpness"])
            N = self._xyz.shape[0]
            self._dc         = _p(t.get("_dc", torch.zeros((N, 3), device="cuda")))
            self.num_lobes   = t.get("num_lobes", self._mu.shape[1])

        self.training_setup(training_args)

        opt_state = data.get("optimizer_state", None)
        if opt_state is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                pass

    def load(self, path, map_location="cuda"):
        """Load saved checkpoint into already-initialised parameters (overwrites tensors
        but does NOT rebuild the optimizer — call training_setup first)."""
        data = torch.load(path, map_location=map_location)
        t = data["tensors"]
        c = data["config"]

        self._xyz         = t["_xyz"]
        self._opacity     = t["_opacity"]
        self._beta        = t["_beta"]
        self._scaling     = t["_scaling"]
        self._rotation    = t["_rotation"]

        self.color_representation = c["color_representation"]
        self.spatial_lr_scale = c.get("spatial_lr_scale", 1.0)

        if self.color_representation in ("sh", "sb"):
            self._sh0        = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number   = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]

        elif self.color_representation == "voronoi":
            self._sh0 = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number  = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]
            self.num_lobes  = t.get("num_lobes", 8)
            self._sites         = nn.Parameter(t["_sites"])
            self._colors        = nn.Parameter(t["_colors"])
            self._log_tau_sites = nn.Parameter(t["_log_tau_sites"])

        elif self.color_representation == "sgs":
            self._sh0 = t["_sh0"]; self._shN = t["_shN"]; self._sb_params = t["_sb_params"]
            self.active_sh_degree = t["active_sh_degree"]
            self.sb_number  = t["sb_number"]; self.max_sh_degree = t["max_sh_degree"]
            self._mu        = t["_mu"]; self._alpha = t["_alpha"]; self._sharpness = t["_sharpness"]
            self._dc        = t.get("_dc", torch.zeros((self._xyz.shape[0], 3), device=map_location))
            self.num_lobes  = t.get("num_lobes", self._mu.shape[1])

        opt_state = data.get("optimizer_state", None)
        if opt_state is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # PLY save / load
    # ------------------------------------------------------------------

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
                      f"color_{li}_r", f"color_{li}_g", f"color_{li}_b",
                      f"tau_{li}"]

        if self.color_representation == "sgs" and hasattr(self, "_mu") and self._mu.numel() > 0:
            L = self._mu.shape[1]
            l += ["dc_r", "dc_g", "dc_b"]
            for li in range(L):
                l += [f"mu_{li}_x", f"mu_{li}_y", f"mu_{li}_z",
                      f"alpha_{li}_r", f"alpha_{li}_g", f"alpha_{li}_b",
                      f"sharp_{li}"]
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz      = self._xyz.detach().cpu().numpy()
        normals  = np.zeros_like(xyz)
        sh0      = self._sh0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        shN      = self._shN.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self._sb_params.numel() > 0:
            sb_params = self._sb_params.transpose(1, 2).detach().flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            sb_params = np.zeros((xyz.shape[0], 0), dtype=np.float32)
        opacities = self._opacity.detach().cpu().numpy()
        betas     = self._beta.detach().cpu().numpy()
        scale     = self._scaling.detach().cpu().numpy()
        rotation  = self._rotation.detach().cpu().numpy()

        extra_cols = []

        if self.color_representation == "voronoi" and hasattr(self, "_sites") and self._sites.numel() > 0:
            sites   = self._sites.detach().cpu().numpy()
            colors  = self._colors.detach().cpu().numpy()
            log_tau = self._log_tau_sites.detach().cpu().numpy()  # [N, L]
            N, L, _ = sites.shape
            for li in range(L):
                extra_cols.append(sites[:, li, :])
                extra_cols.append(colors[:, li, :])
                extra_cols.append(log_tau[:, li: li + 1])

        if self.color_representation == "sgs" and hasattr(self, "_mu") and self._mu.numel() > 0:
            mu    = self._mu.detach().cpu().numpy()
            alpha = self._alpha.detach().cpu().numpy()
            sharp = self._sharpness.detach().cpu().numpy()
            dc    = self._dc.detach().cpu().numpy()
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
        assert attributes.shape[1] == len(dtype_full), \
            f"Mismatch: attributes {attributes.shape[1]} vs dtype {len(dtype_full)}"
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata  = PlyData.read(path)
        elem     = plydata.elements[0]
        xyz      = np.stack((np.asarray(elem["x"]), np.asarray(elem["y"]), np.asarray(elem["z"])), axis=1)
        opacities = np.asarray(elem["opacity"])[..., np.newaxis]
        betas    = np.asarray(elem["beta"])[..., np.newaxis]

        sh0 = np.zeros((xyz.shape[0], 3, 1))
        sh0[:, 0, 0] = np.asarray(elem["sh0_0"])
        sh0[:, 1, 0] = np.asarray(elem["sh0_1"])
        sh0[:, 2, 0] = np.asarray(elem["sh0_2"])

        prop_names = [p.name for p in elem.properties]
        shN_names = sorted(
            [n for n in prop_names if n.startswith("shN_")],
            key=lambda x: int(x.split("_")[-1])
        )
        assert len(shN_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        shs_extra = np.zeros((xyz.shape[0], len(shN_names)))
        for idx, attr_name in enumerate(shN_names):
            shs_extra[:, idx] = np.asarray(elem[attr_name])
        shs_extra = shs_extra.reshape(
            (shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        sb_names = sorted(
            [n for n in prop_names if n.startswith("sb_params_")],
            key=lambda x: int(x.split("_")[-1])
        )
        if self.sb_number > 0:
            assert len(sb_names) == self.sb_number * 6
            sb_params = np.zeros((xyz.shape[0], len(sb_names)))
            for idx, attr_name in enumerate(sb_names):
                sb_params[:, idx] = np.asarray(elem[attr_name])
            sb_params = sb_params.reshape((sb_params.shape[0], 6, self.sb_number))
        else:
            sb_params = np.zeros((xyz.shape[0], 6, 0))

        scale_names = sorted(
            [n for n in prop_names if n.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1])
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(elem[attr_name])

        rot_names = sorted(
            [n for n in prop_names if n.startswith("rot_")],
            key=lambda x: int(x.split("_")[-1])
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(elem[attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sh0 = nn.Parameter(
            torch.tensor(sh0, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._shN = nn.Parameter(
            torch.tensor(shs_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._sb_params = nn.Parameter(
            torch.tensor(sb_params, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._beta = nn.Parameter(
            torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree

        # --- voronoi ---
        if self.color_representation == "voronoi":
            site_xyz_names = [
                n for n in prop_names
                if n.startswith("site_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))
            ]
            color_names = [
                n for n in prop_names
                if n.startswith("color_") and (n.endswith("_r") or n.endswith("_g") or n.endswith("_b"))
            ]
            tau_names = sorted(
                [n for n in prop_names if n.startswith("tau_")],
                key=lambda x: int(x.split("_")[1])
            )
            if len(site_xyz_names) > 0:
                def site_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]

                def color_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"r": 0, "g": 1, "b": 2}[parts[2]]

                site_xyz_names = sorted(site_xyz_names, key=site_key)
                color_names    = sorted(color_names, key=color_key)

                sites_flat = np.zeros((xyz.shape[0], len(site_xyz_names)))
                for idx, attr_name in enumerate(site_xyz_names):
                    sites_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(site_xyz_names) // 3
                sites = sites_flat.reshape(xyz.shape[0], L, 3)

                colors_flat = np.zeros((xyz.shape[0], len(color_names)))
                for idx, attr_name in enumerate(color_names):
                    colors_flat[:, idx] = np.asarray(elem[attr_name])
                colors = colors_flat.reshape(xyz.shape[0], L, 3)

                if len(tau_names) > 0:
                    tau_flat = np.zeros((xyz.shape[0], len(tau_names)))
                    for idx, attr_name in enumerate(tau_names):
                        tau_flat[:, idx] = np.asarray(elem[attr_name])
                else:
                    tau_flat = np.zeros((xyz.shape[0], L), dtype=np.float32)

                self._sites = nn.Parameter(
                    torch.tensor(sites, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._colors = nn.Parameter(
                    torch.tensor(colors, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._log_tau_sites = nn.Parameter(
                    torch.tensor(tau_flat, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self.num_lobes = L

        # --- sgs ---
        if self.color_representation == "sgs":
            dc = None
            if all(n in prop_names for n in ["dc_r", "dc_g", "dc_b"]):
                dc = np.stack(
                    (np.asarray(elem["dc_r"]), np.asarray(elem["dc_g"]), np.asarray(elem["dc_b"])),
                    axis=1,
                )
            mu_names    = [n for n in prop_names if n.startswith("mu_") and (n.endswith("_x") or n.endswith("_y") or n.endswith("_z"))]
            alpha_names = [n for n in prop_names if n.startswith("alpha_") and (n.endswith("_r") or n.endswith("_g") or n.endswith("_b"))]
            sharp_names = [n for n in prop_names if n.startswith("sharp_")]
            if len(mu_names) > 0:
                def mu_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"x": 0, "y": 1, "z": 2}[parts[2]]

                def alpha_key(name):
                    parts = name.split("_")
                    return int(parts[1]) * 3 + {"r": 0, "g": 1, "b": 2}[parts[2]]

                mu_names    = sorted(mu_names, key=mu_key)
                alpha_names = sorted(alpha_names, key=alpha_key)
                sharp_names = sorted(sharp_names, key=lambda x: int(x.split("_")[1]))

                mu_flat = np.zeros((xyz.shape[0], len(mu_names)))
                for idx, attr_name in enumerate(mu_names):
                    mu_flat[:, idx] = np.asarray(elem[attr_name])
                L = len(mu_names) // 3
                mu = mu_flat.reshape(xyz.shape[0], L, 3)

                alpha_flat = np.zeros((xyz.shape[0], len(alpha_names)))
                for idx, attr_name in enumerate(alpha_names):
                    alpha_flat[:, idx] = np.asarray(elem[attr_name])
                alpha = alpha_flat.reshape(xyz.shape[0], L, 3)

                sharp_flat = np.zeros((xyz.shape[0], len(sharp_names)))
                for idx, attr_name in enumerate(sharp_names):
                    sharp_flat[:, idx] = np.asarray(elem[attr_name])
                sharp = sharp_flat.reshape(xyz.shape[0], L, 1)

                self._mu = nn.Parameter(
                    torch.tensor(mu, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._alpha = nn.Parameter(
                    torch.tensor(alpha, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                self._sharpness = nn.Parameter(
                    torch.tensor(sharp, dtype=torch.float, device="cuda").requires_grad_(True)
                )
                if dc is not None:
                    self._dc = nn.Parameter(
                        torch.tensor(dc, dtype=torch.float, device="cuda").requires_grad_(True)
                    )
                self.num_lobes = L

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def training_setup(self, training_args):
        l = [
            {"params": [self._xyz],      "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._opacity],  "lr": training_args.opacity_lr,  "name": "opacity"},
            {"params": [self._beta],     "lr": training_args.beta_lr,     "name": "beta"},
            {"params": [self._scaling],  "lr": training_args.scaling_lr,  "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if self.color_representation in ("sh", "sb"):
            l += [
                {"params": [self._sh0],       "lr": training_args.sh_lr,            "name": "sh0"},
                {"params": [self._shN],       "lr": training_args.sh_lr / 20.0,     "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr,     "name": "sb_params"},
            ]

        elif self.color_representation == "voronoi":
            sites_lr = training_args.sites_lr
            color_lr = training_args.color_lr
            tau_lr   = getattr(training_args, "tau_lr", sites_lr)
            self.voronoi_sites_lr_init = sites_lr
            self.voronoi_tau_lr_init   = tau_lr
            self.total_iterations = getattr(training_args, "iterations", 45000)
            l += [
                {"params": [self._sh0],            "lr": training_args.sh_lr,        "name": "sh0"},
                {"params": [self._shN],            "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params],      "lr": training_args.sb_params_lr, "name": "sb_params"},
                {"params": [self._sites],          "lr": sites_lr,                   "name": "sites"},
                {"params": [self._colors],         "lr": color_lr,                   "name": "colors"},
                {"params": [self._log_tau_sites],  "lr": tau_lr,                     "name": "log_tau_sites"},
            ]

        elif self.color_representation == "sgs":
            l += [
                {"params": [self._sh0],       "lr": training_args.sh_lr,        "name": "sh0"},
                {"params": [self._shN],       "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {"params": [self._sb_params], "lr": training_args.sb_params_lr, "name": "sb_params"},
                {"params": [self._mu],        "lr": training_args.sites_lr,     "name": "mu"},
                {"params": [self._alpha],     "lr": training_args.color_lr,     "name": "alpha"},
                {"params": [self._sharpness], "lr": 1e-2,                       "name": "sharpness"},
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
            max_steps=30_000,
        )

    def update_learning_rate(self, iteration):
        xyz_lr = None
        _voronoi_frozen = (
            self.color_representation == "voronoi"
            and hasattr(self, "total_iterations")
            and iteration > self.total_iterations - 3000
        )
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                if not _voronoi_frozen:
                    lr = self.xyz_scheduler_args(iteration)
                    param_group["lr"] = lr
                xyz_lr = param_group["lr"]

        if self.color_representation == "voronoi" and hasattr(self, "voronoi_sites_lr_init"):
            if iteration <= self.total_iterations - 3000:
                T = self.total_iterations
                t = math.pi * iteration / T

                lr0_sites = self.voronoi_sites_lr_init
                cos_lr_sites = lr0_sites * 0.1 + 0.5 * lr0_sites * 0.9 * (1 + math.cos(t))

                lr0_tau = self.voronoi_tau_lr_init
                cos_lr_tau = lr0_tau * 0.1 + 0.5 * lr0_tau * 0.9 * (1 + math.cos(t))

                for pg in self.optimizer.param_groups:
                    if pg["name"] == "sites":
                        pg["lr"] = cos_lr_sites
                    elif pg["name"] == "log_tau_sites":
                        pg["lr"] = cos_lr_tau
        return xyz_lr

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"]    = torch.zeros_like(tensor)
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
                stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
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
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, params, reset_params=True):
        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._xyz        = optimizable_tensors["xyz"]
        self._sh0        = optimizable_tensors["sh0"]
        self._shN        = optimizable_tensors["shN"]
        self._sb_params  = optimizable_tensors["sb_params"]
        self._opacity    = optimizable_tensors["opacity"]
        self._beta       = optimizable_tensors["beta"]
        self._scaling    = optimizable_tensors["scaling"]
        self._rotation   = optimizable_tensors["rotation"]

        if self.color_representation == "voronoi":
            self._sites         = optimizable_tensors["sites"]
            self._colors        = optimizable_tensors["colors"]
            self._log_tau_sites = optimizable_tensors["log_tau_sites"]
        elif self.color_representation == "sgs":
            self._mu         = optimizable_tensors["mu"]
            self._alpha      = optimizable_tensors["alpha"]
            self._sharpness  = optimizable_tensors["sharpness"]

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz, "sh0": self._sh0, "shN": self._shN,
            "sb_params": self._sb_params, "opacity": self._opacity,
            "beta": self._beta, "scaling": self._scaling, "rotation": self._rotation,
        }
        if self.color_representation == "voronoi":
            tensors_dict.update({
                "sites": self._sites, "colors": self._colors,
                "log_tau_sites": self._log_tau_sites,
            })
        elif self.color_representation == "sgs":
            tensors_dict.update({
                "mu": self._mu, "alpha": self._alpha, "sharpness": self._sharpness,
            })

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
                    stored_state["exp_avg"][inds]    = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"]    = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz        = optimizable_tensors["xyz"]
        self._sh0        = optimizable_tensors["sh0"]
        self._shN        = optimizable_tensors["shN"]
        self._sb_params  = optimizable_tensors["sb_params"]
        self._opacity    = optimizable_tensors["opacity"]
        self._beta       = optimizable_tensors["beta"]
        self._scaling    = optimizable_tensors["scaling"]
        self._rotation   = optimizable_tensors["rotation"]

        if self.color_representation == "voronoi":
            self._sites         = optimizable_tensors["sites"]
            self._colors        = optimizable_tensors["colors"]
            self._log_tau_sites = optimizable_tensors["log_tau_sites"]
        elif self.color_representation == "sgs":
            self._mu        = optimizable_tensors["mu"]
            self._alpha     = optimizable_tensors["alpha"]
            self._sharpness = optimizable_tensors["sharpness"]

        torch.cuda.empty_cache()
        return optimizable_tensors

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------

    def prune(self, live_mask):
        self._xyz        = self._xyz[live_mask]
        self._sh0        = self._sh0[live_mask]
        self._shN        = self._shN[live_mask]
        self._sb_params  = self._sb_params[live_mask]
        self._scaling    = self._scaling[live_mask]
        self._rotation   = self._rotation[live_mask]
        self._opacity    = self._opacity[live_mask]
        self._beta       = self._beta[live_mask]
        if self.color_representation == "voronoi":
            self._sites         = self._sites[live_mask]
            self._colors        = self._colors[live_mask]
            self._log_tau_sites = self._log_tau_sites[live_mask]
        elif self.color_representation == "sgs":
            self._mu        = self._mu[live_mask]
            self._alpha     = self._alpha[live_mask]
            self._sharpness = self._sharpness[live_mask]
            self._dc        = self._dc[live_mask]

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1))
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps, min=0.005
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        ret = [
            self._xyz[idxs], self._sh0[idxs], self._shN[idxs],
            self._sb_params[idxs], new_opacity, self._beta[idxs],
            self._scaling[idxs], self._rotation[idxs],
        ]
        if self.color_representation == "voronoi":
            ret += [self._sites[idxs], self._colors[idxs], self._log_tau_sites[idxs]]
        elif self.color_representation == "sgs":
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
        alive_mask    = ~dead_mask
        dead_indices  = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )
        (self._xyz[dead_indices], self._sh0[dead_indices], self._shN[dead_indices],
         self._sb_params[dead_indices], self._opacity[dead_indices], self._beta[dead_indices],
         self._scaling[dead_indices], self._rotation[dead_indices], *rest) = \
            self._update_params(reinit_idx, ratio=ratio)

        if self.color_representation == "voronoi":
            self._sites[dead_indices]         = rest[0]
            self._colors[dead_indices]        = rest[1]
            self._log_tau_sites[dead_indices] = rest[2]
        elif self.color_representation == "sgs":
            self._mu[dead_indices]        = rest[0]
            self._alpha[dead_indices]     = rest[1]
            self._sharpness[dead_indices] = rest[2]

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def subsample_to(self, cap_max):
        N = self._xyz.shape[0]
        if N <= cap_max:
            return
        perm = torch.randperm(N, device='cuda')[:cap_max]
        mask = torch.zeros(N, dtype=torch.bool, device='cuda')
        mask[perm] = True

        optimizable_tensors = self._prune_optimizer(mask)
        self._xyz        = optimizable_tensors["xyz"]
        self._sh0        = optimizable_tensors["sh0"]
        self._shN        = optimizable_tensors["shN"]
        self._sb_params  = optimizable_tensors["sb_params"]
        self._opacity    = optimizable_tensors["opacity"]
        self._beta       = optimizable_tensors["beta"]
        self._scaling    = optimizable_tensors["scaling"]
        self._rotation   = optimizable_tensors["rotation"]

        if self.color_representation == "voronoi":
            self._sites         = optimizable_tensors["sites"]
            self._colors        = optimizable_tensors["colors"]
            self._log_tau_sites = optimizable_tensors["log_tau_sites"]
        elif self.color_representation == "sgs":
            self._mu        = optimizable_tensors["mu"]
            self._alpha     = optimizable_tensors["alpha"]
            self._sharpness = optimizable_tensors["sharpness"]


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

        params = {
            "xyz": new_xyz, "opacity": new_opacity, "scaling": new_scaling,
            "rotation": new_rotation, "sh0": new_sh0, "shN": new_shN,
            "sb_params": new_sb_params, "beta": new_beta,
        }
        if self.color_representation == "voronoi":
            params.update({"sites": rest[0], "colors": rest[1], "log_tau_sites": rest[2]})
        elif self.color_representation == "sgs":
            params.update({"mu": rest[0], "alpha": rest[1], "sharpness": rest[2]})

        self._opacity[add_idx] = new_opacity
        self.densification_postfix(params, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, viewpoint_camera, render_mode="RGB"):

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)
        fx = 0.5 * viewpoint_camera.image_width  / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)
        K[0, 0] = fx; K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width  / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0


        if self.color_representation in ("voronoi", "sgs"):
            dir_pp = (
                self.get_xyz
                - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1)
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = self.compute_radiance(dir_pp_normalized)
            active_degree  = None
        else:  # sh or sb
            colors_precomp = self.get_shs
            active_degree  = self.active_sh_degree

        rgbs, alphas, meta = rasterization(
            means=self.get_xyz, quats=self.get_rotation,
            scales=self.get_scaling, opacities=self.get_opacity.squeeze(),
            betas=self.get_beta.squeeze(), colors=colors_precomp,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0), width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode, covars=None, sh_degree=active_degree,
            sb_number=self.sb_number, sb_params=self.get_sb_params, packed=False,
        )
        rgb = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        return {
            "render":            rgb,
            "viewspace_points":  meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii":             meta["radii"],
            "is_used":           meta["radii"] > 0,
        }
