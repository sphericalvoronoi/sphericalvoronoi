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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import torch.nn.functional as F
import nvdiffrast.torch as dr
from pytorch3d.ops import knn_points, knn_gather
# import faiss

def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** ((x - 1).bit_length())

@torch.no_grad()
def init_cubemap_atlas_from_grid(self, grid_map6: torch.Tensor, pad: int = 4, make_pow2: bool = True):
    """
    Inizializza:
      - self.cube_atlas: nn.Parameter [1,H,W,C] (ottimizzabile)
      - self.cube_tiles: Tensor [V,6,4] (ox,oy,sx,sy) in [0,1] (STATICO)
      - self.max_lod_tile: int   (LOD max sensato per il TILE, non per l'intero atlas)
      - self.mip_levels:   int|None (compatibile con dr.texture; impostato = self.max_lod_tile)
    """
    assert grid_map6.dim() == 5, f"atteso [V,6,R,R,C], trovato {grid_map6.shape}"
    V, six, R, R2, C = grid_map6.shape
    assert six == 6 and R == R2 and C in (3,4)
    dev, dt = grid_map6.device, grid_map6.dtype

    # layout a griglia quadrata di tutte le facce
    num_tiles = V * 6
    tiles_per_row = math.ceil(math.sqrt(num_tiles)) if num_tiles > 0 else 1
    tile_px = int(R + 2 * pad)                 # <--- dimensione del TILE (faccia + padding)
    base_atlas = tiles_per_row * tile_px
    atlas_px = _next_pow2(base_atlas) if make_pow2 else base_atlas
    H = W = int(atlas_px)

    # atlas ottimizzabile
    atlas = torch.zeros((1, H, W, C), device=dev, dtype=dt)

    # tabella mapping (area utile senza pad): [V,6,4]
    cube_tiles = torch.empty((V, 6, 4), device=dev, dtype=torch.float32)

    t = 0
    for v in range(V):
        for f in range(6):
            row, col = divmod(t, tiles_per_row)
            x0, y0 = col * tile_px, row * tile_px
            xs, xe = x0, x0 + tile_px
            ys, ye = y0, y0 + tile_px

            face_tex = grid_map6[v, f]  # [R,R,C]

            # centro
            atlas[0, ys+pad:ys+pad+R, xs+pad:xs+pad+R, :] = face_tex
            # bordi replicati
            atlas[0, ys:ys+pad,          xs+pad:xs+pad+R, :] = face_tex[0:1].expand(pad, R, C)      # top
            atlas[0, ys+pad+R:ye,        xs+pad:xs+pad+R, :] = face_tex[-1:].expand(pad, R, C)      # bottom
            atlas[0, ys+pad:ys+pad+R,    xs:xs+pad,          :] = face_tex[:, 0:1, :].expand(R, pad, C)  # left
            atlas[0, ys+pad:ys+pad+R,    xs+pad+R:xe,        :] = face_tex[:, -1:, :].expand(R, pad, C)  # right
            # angoli
            atlas[0, ys:ys+pad,          xs:xs+pad,          :] = face_tex[0,0].view(1,1,C).expand(pad,pad,C)
            atlas[0, ys:ys+pad,          xs+pad+R:xe,        :] = face_tex[0,-1].view(1,1,C).expand(pad,pad,C)
            atlas[0, ys+pad+R:ye,        xs:xs+pad,          :] = face_tex[-1,0].view(1,1,C).expand(pad,pad,C)
            atlas[0, ys+pad+R:ye,        xs+pad+R:xe,        :] = face_tex[-1,-1].view(1,1,C).expand(pad,pad,C)

            # offset/scale NORMALIZZATI per area utile (senza pad)
            ox = (xs + pad) / W
            oy = (ys + pad) / H
            sx = R / W
            sy = R / H
            cube_tiles[v, f] = torch.tensor([ox, oy, sx, sy], device=dev)

            t += 1

    # salva come param e mapping statico
    self.cube_atlas = nn.Parameter(atlas)   # ottimizzabile direttamente
    self.cube_tiles = cube_tiles            # statico

    # --- SALVA LOD PER TILE (NON per l'intero atlas!) ---
    self.pad_px  = int(pad)
    self.map_res = int(R)                   # se non lo avevi già
    self.tile_px = int(self.map_res + 2 * self.pad_px)       # es. 48 + 8 = 56
    self.max_lod_tile = int(torch.floor(torch.log2(torch.tensor(self.tile_px, dtype=torch.float32))).item())
    # compat: usa questo anche come 'mip_levels' se lo passi a dr.texture
    self.mip_levels = self.max_lod_tile

            

def dirs_to_cubemap_uv01(dirs: torch.Tensor):
    x, y, z = dirs.unbind(-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()

    is_x = (ax >= ay) & (ax >= az)
    is_y = (~is_x) & (ay >= az)

    face = torch.empty_like(x, dtype=torch.int64)
    u = torch.empty_like(x)
    v = torch.empty_like(x)

    # +X
    m = is_x & (x >= 0)
    ma = ax.clamp_min(1e-8)
    u[m] = (-z[m] / ma[m]); v[m] = (-y[m] / ma[m]); face[m] = 0
    # -X
    m = is_x & (x < 0)
    ma = ax.clamp_min(1e-8)
    u[m] = ( z[m] / ma[m]); v[m] = (-y[m] / ma[m]); face[m] = 1
    # +Y
    m = is_y & (y >= 0)
    ma = ay.clamp_min(1e-8)
    u[m] = ( x[m] / ma[m]); v[m] = ( z[m] / ma[m]); face[m] = 2
    # -Y
    m = is_y & (y < 0)
    ma = ay.clamp_min(1e-8)
    u[m] = ( x[m] / ma[m]); v[m] = (-z[m] / ma[m]); face[m] = 3
    # +Z
    m = (~is_x) & (~is_y) & (z >= 0)
    ma = az.clamp_min(1e-8)
    u[m] = ( x[m] / ma[m]); v[m] = (-y[m] / ma[m]); face[m] = 4
    # -Z
    m = (~is_x) & (~is_y) & (z < 0)
    ma = az.clamp_min(1e-8)
    u[m] = (-x[m] / ma[m]); v[m] = (-y[m] / ma[m]); face[m] = 5

    u01 = u.mul_(0.5).add_(0.5)
    v01 = v.mul_(0.5).add_(0.5)
    return face, torch.stack([u01, v01], dim=-1)  # (N,), (N,2)


def dir_to_face_uv(d):  # d: (B,3), unit
    x, y, z = d[:,0], d[:,1], d[:,2]
    ax, ay, az = x.abs(), y.abs(), z.abs()

    cond_x = (ax >= ay) & (ax >= az)
    cond_y = (ay >  ax) & (ay >= az)
    cond_z = ~(cond_x | cond_y)

    face = torch.empty_like(x, dtype=torch.long)
    u = torch.empty_like(x, dtype=d.dtype)
    v = torch.empty_like(x, dtype=d.dtype)

    mask = cond_x & (x > 0);  u[mask] = (-z[mask] / ax[mask]); v[mask] = (-y[mask] / ax[mask]); face[mask] = 0
    mask = cond_x & (x <= 0); u[mask] = ( z[mask] / ax[mask]); v[mask] = (-y[mask] / ax[mask]); face[mask] = 1
    mask = cond_y & (y > 0);  u[mask] = ( x[mask] / ay[mask]); v[mask] = ( z[mask] / ay[mask]); face[mask] = 2
    mask = cond_y & (y <= 0); u[mask] = ( x[mask] / ay[mask]); v[mask] = (-z[mask] / ay[mask]); face[mask] = 3
    mask = cond_z & (z > 0);  u[mask] = ( x[mask] / az[mask]); v[mask] = (-y[mask] / az[mask]); face[mask] = 4
    mask = cond_z & (z <= 0); u[mask] = (-x[mask] / az[mask]); v[mask] = (-y[mask] / az[mask]); face[mask] = 5

    return face, u.clamp(-1,1), v.clamp(-1,1)

def uv_to_ij(u, v, R):
    i = torch.clamp(((u*0.5 + 0.5) * R).floor().long(), 0, R-1)
    j = torch.clamp(((v*0.5 + 0.5) * R).floor().long(), 0, R-1)
    return i, j


def texel_centers_dirs(R, device='cuda', dtype=torch.float32):
    # Build all centers for 6 faces at once (C,3), C=6*R*R
    rr = torch.arange(R, device=device)
    jg, ig = torch.meshgrid(rr, rr, indexing='ij')
    u = ((ig.float()+0.5)/R)*2-1
    v = ((jg.float()+0.5)/R)*2-1

    ones = torch.ones_like(u)
    # +X, -X, +Y, -Y, +Z, -Z
    d0 = torch.stack([ ones, -v,   -u], dim=-1)  # +X
    d1 = torch.stack([-ones, -v,    u], dim=-1)  # -X
    d2 = torch.stack([   u ,  ones,  v], dim=-1) # +Y
    d3 = torch.stack([   u , -ones, -v], dim=-1) # -Y
    d4 = torch.stack([   u ,  -v ,  ones], dim=-1) # +Z
    d5 = torch.stack([  -u ,  -v , -ones], dim=-1) # -Z

    D = torch.cat([d0, d1, d2, d3, d4, d5], dim=0).reshape(-1,3)
    return F.normalize(D, dim=-1).to(device=device, dtype=dtype)


def face_ij_to_cellid(face, i, j, R):
    return (face*R + j)*R + i  # (B,)


def positional_encoding(x: torch.Tensor, num_freqs: int = 4, include_input: bool = True) -> torch.Tensor:

    assert x.ndim == 2
    device = x.device

    if num_freqs == 0:
        return x if include_input else torch.empty(x.shape[0], 0, device=device)

    freqs = 2.0 ** torch.arange(num_freqs, device=device).float()  # (num_freqs,)
    x_freq = x.unsqueeze(-1) * freqs  # (N, C, num_freqs)

    x_sin = torch.sin(x_freq)  # (N, C, num_freqs)
    x_cos = torch.cos(x_freq)  # (N, C, num_freqs)

    x_encoded = torch.cat([x_sin, x_cos], dim=-1)  # (N, C, 2 * num_freqs)
    x_encoded = x_encoded.view(x.shape[0], -1)     # (N, C * 2 * num_freqs)

    if include_input:
        return torch.cat([x, x_encoded], dim=-1)
    else:
        return x_encoded



def fibonacci_sphere(K, radius=1.0, center=None, device=None, dtype=torch.float32):
    i = torch.arange(K, dtype=dtype, device=device)
    phi = (1 + 5**0.5) / 2
    z = 1 - 2*(i+0.5)/K
    r = torch.sqrt(torch.clamp(1 - z*z, min=0))
    theta = 2*math.pi*i/phi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1)
    pts = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pts = pts.cuda() * radius
    if center is not None:
        center = torch.as_tensor(center, dtype=dtype, device=device)
        pts = pts + center.cpu()
    return pts  # (K,3)


import torch
import torch.nn as nn
import math

def kaiming_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # ReLU
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)





def init_mlp_sites_and_color(mlp_sites, mlp_color, K, r0=1, device='cuda'):
    """ 
    - Inizializza hidden con Kaiming
    - Ultimo layer pesi ~0
    - Bias ultimo layer sites = ancore direzionali uniformi * r0
    - Bias ultimo layer color = 0 (o piccolo)
    """
    # 1) He init su tutto
    mlp_sites.apply(kaiming_init_)
    mlp_color.apply(kaiming_init_)

    # 2) individua l'ultimo Linear di ciascun MLP
    def last_linear(module):
        layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        return layers[-1]

    last_s = last_linear(mlp_sites)
    last_c = last_linear(mlp_color)

    # 3) Pesi finali piccoli → output iniziale ≈ bias
    nn.init.zeros_(last_s.weight)
    nn.init.zeros_(last_c.weight)

    # 4) Bias sites = K direzioni uniformi * r0, ripiegate in [K*3]
    anchors = fibonacci_sphere(K).to(device) * r0  # (K,3)
    last_s.bias.data = anchors.reshape(-1).to(device)

    # 5) Bias color: zero (o leggero neutro)
    nn.init.constant_(mlp_color[-1].bias, 0.2)



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans #trans[:,:3,3]
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, color_representation='shs', degree=1, map_res=32, 
                 use_mlp=True, num_probes=8, num_sites=2048, nn_k=8, topk=8):
        self.active_sh_degree = 0
        self.max_sh_degree = 3  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.color_representation = color_representation
        self.feat_dim = 16
        self.num_pos_freq = 4
        self.use_mlp = False
        self.nn_k = nn_k
        self.topk = topk
        if color_representation == 'sgs':
            self._mu = torch.empty(0)
            self._alpha = torch.empty(0)
            self._lambda = torch.empty(0)
            self._dc = torch.empty(0)
            self.max_degree = degree
            self.active_degree = 0
        elif color_representation == 'voronoi':
            self._sites = torch.empty(0)
            self._beta = torch.empty(0)
            self._lambd = torch.empty(0)
            self.max_degree = degree
            self.active_degree = 0
            
        elif color_representation == 'big_voronoi':
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            
        elif color_representation == 'voronoi_r':
            self._sites = torch.empty(0)
            self._beta = torch.empty(0)
            self._lambd = torch.empty(0)
            self.max_degree = degree
            self.active_degree = 0
            self._dc = torch.empty(0)
            self._roughness = torch.empty(0)
            
        elif color_representation == 'lp_voronoi_cmap':
            self.G = 2
            self.K = degree
            self.map_res = 128
            self.V = self.G**3
            self.topk = 24
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self._kappa = torch.empty(0)
            self.centers = octa_texel_centers_dirs(self.map_res, device='cuda')
            
        elif self.color_representation == 'lp_opt_grid':
            self.G = 8
            self.color_dim = 3
            self.map_res = map_res
            self.V = self.G ** 3
            self.K = num_sites
            self.centers = texel_centers_dirs(self.map_res, device='cuda')
            self.bbox_min = torch.empty(0)
            self.bbox_max = torch.empty(0)
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self._alpha = torch.empty(0)
            self._roughness = torch.empty(0)
            self.bbox_min = torch.empty(0)
            self.bbox_max = torch.empty(0)
            self.map_res = map_res
            self.mip_levels = 9
            
        elif self.color_representation == 'lp_opt' or self.color_representation == 'lp_opt_d':
            self.G = num_probes
            self.color_dim = 3
            self.map_res = map_res
            self.V = num_probes
            self.K = num_sites
            self.centers = texel_centers_dirs(self.map_res, device='cuda')
            self.bbox_min = torch.empty(0)
            self.bbox_max = torch.empty(0)
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self._alpha = torch.empty(0)
            self._roughness = torch.empty(0)
            self.bbox_min = torch.empty(0)
            self.bbox_max = torch.empty(0)
            self.map_res = map_res
            self.mip_levels = 9
            self.num_channels = 3
            self.hidden_dim = 128
            self.map = nn.Parameter(torch.full((1, 6, 256, 256, self.num_channels), fill_value=0.25, device='cuda'))
            self.voronoi_env = False
            
        elif color_representation == 'cubemap':
            self.map_res = map_res
            self.mip_levels = 9
            self.num_channels = 3
            self.hidden_dim = 128
            self.map = nn.Parameter(torch.full((1, 6, 256, 256, self.num_channels), fill_value=0.25, device='cuda'))
            #self.map = nn.Parameter(torch.zeros((1, 6, self.map_res, self.map_res, self.num_channels), device='cuda') - 1.25)
            
            self.mlp_decoder = nn.Sequential (
                    nn.Linear(self.num_channels, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 3)).to('cuda')
        
             
        elif color_representation == 'light_probes':
            self._dc = torch.empty(0)
            
            self._roughness = torch.empty(0)
            self.bbox_min = torch.empty(0)
            self.bbox_max = torch.empty(0)
            self.max_degree = degree
            self.res = 5
            self._sites = torch.empty(0)
            self._colors = torch.empty(0)
            self.use_light_probes = True
            self.use_mlp = use_mlp
            hidden_dim = 256
            num_hidden_layers = 2
            self.num_pos_freq = 8
            self.feat_dim = 16
            self.topk = 256
            self._betas = torch.empty(0)
            self.num_feat_colors = 3
            
            self.mlp_sites = nn.Sequential(
                nn.Linear(self.feat_dim + self.num_pos_freq*2*3+3, hidden_dim),
                nn.ReLU(True),
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, self.max_degree*3)
            ).to('cuda')

            self.mlp_color = nn.Sequential(
                nn.Linear(self.feat_dim + self.num_pos_freq*2*3+3, hidden_dim), 
                nn.ReLU(True),
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, self.max_degree*self.num_feat_colors)
            ).to('cuda')
            
            
            self.mlp_kappa = nn.Sequential(
                nn.Linear(self.feat_dim + self.num_pos_freq*2*3+3, hidden_dim), 
                nn.ReLU(True),
                *[
                    nn.Sequential(nn.Linear(hidden_dim , hidden_dim), nn.ReLU(True))
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, self.max_degree)
            ).to('cuda')
            
            # nn.init.constant_(self.mlp_color[-1].bias, 0.5)
            # nn.init.constant_(self.mlp_sites[-1].bias, 0.5)
            
            
            # self.mlp_sites = nn.Sequential(
            #     nn.Linear(3 + 4, hidden_dim),
            #     nn.ReLU(True),
            #     *[
            #         nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))
            #         for _ in range(num_hidden_layers - 1)
            #     ],
            #     nn.Linear(hidden_dim, self.max_degree*3)
            # ).to('cuda')

            # self.mlp_color = nn.Sequential(
            #     nn.Linear(3 + 4, hidden_dim), 
            #     nn.ReLU(True),
            #     *[
            #         nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))
            #         for _ in range(num_hidden_layers - 1)
            #     ],
            #     nn.Linear(hidden_dim, self.max_degree*3)
            # ).to('cuda')
            
            self.light_mlp = nn.Sequential(
                nn.Linear(self.feat_dim + self.num_pos_freq*2*3+3 + 1, hidden_dim), 
                nn.ReLU(True),
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, 3)
            ).to('cuda')
            
            
            self.decoder = nn.Sequential (
                    nn.Linear(self.num_feat_colors, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3)).to('cuda')
        
        self.setup_functions()
        
        #init_mlp_sites_and_color(self.mlp_sites, self.mlp_color, self.max_degree)


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
    
    @property
    def get_mask(self):
        return torch.sigmoid(self._mask)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property 
    def get_roughness(self):
        return  torch.sigmoid(self._roughness)
    
    
    @property
    def get_albedo(self):
        # bias = torch.tensor(5.0, dtype=torch.float32).to("cuda")
        # return torch.exp(torch.clamp(self._dc, max=5.0))
        return self._dc
    
    
    @property
    def get_refl(self):
        return torch.sigmoid(self._mask)
    
    
    @property
    def get_language_features(self):
        return torch.sigmoid(self._language_features)
    
    
    def predict_normals(self, xyz):
        inp = positional_encoding(xyz, self.num_pos_freq)
        return self.mlp_normals(inp)


    def compute_specular_mlp(self, wo, pos, roughness, feautre_map=None):
        rel = (pos - self.bbox_min) / (self.bbox_max - self.bbox_min)
        coords = rel * 2 - 1
        grid = coords[:, [2, 1, 0]].view(1, pos.shape[0], 1, 1, 3)
        feats = self._features
        
        interp_feat = F.grid_sample(feats, grid, mode='bilinear', align_corners=True).view(self.feat_dim, pos.shape[0]).permute(1, 0)
        wo = positional_encoding(wo, num_freqs=self.num_pos_freq)
        inp = torch.cat([interp_feat, wo, roughness], dim=-1)
        specular = F.softplus(self.light_mlp(inp))
        return specular
    
    
    @torch.no_grad()
    def init_probes(self, ENV_CENTER=None, ENV_RADIUS=None):
        def get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS):
            return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        
        if ENV_CENTER is not None:
            gs_mask = self.get_mask
            gs_in = gs_mask * 0 + 1.0
            gs_in[get_outside_msk(self.get_xyz, ENV_CENTER, ENV_RADIUS)] = 0.0
            
            in_pos = self.get_xyz[gs_in.bool().squeeze(-1)]
            
            self.bbox_min = in_pos.min(dim=0)[0]
            self.bbox_max = in_pos.max(dim=0)[0]
        else:
            self.bbox_max = self._xyz.min(dim=0)[0]
            self.bbox_max = self._xyz.max(dim=0)[1]
            
        # points = self.bbox_min + (self.bbox_max - self.bbox_min) * torch.rand(self.V, 3, device='cuda')
        
        # self.positions._data = points.data
        
        # colors = self.eval_envmap(self._sites.reshape(-1, 3)).reshape(self.V, -1, 3)
        # colors = colors + torch.rand_like(colors) * 1e-3
        # self._colors.data.copy_(colors)
        

    def eval_envmap(self, wo, roughness=None):
        """
        wo: [N,3] (direzioni, su device di wo, idealmente normalizzate)
        roughness: None oppure [N,1] in [0,1]
        Ritorna: [N,C]
        """
        tex = self.map   # [1,6,Hf,Wf,C]
        C = tex.shape[-1]
        device = wo.device
        dtype  = wo.dtype

        # --- griglia quasi quadrata
        N = wo.shape[0]
        Wimg = int(math.ceil(math.sqrt(N)))
        Himg = int(math.ceil(N / Wimg))
        total = Himg * Wimg

        # --- padding direzioni (poi tagliato)
        if total != N:
            pad = total - N
            pad_dirs = torch.zeros(pad, 3, device=device, dtype=dtype)
            pad_dirs[:, 2] = 1.0  # +Z
            wo = torch.cat([wo, pad_dirs], dim=0)

        wo_img = wo.view(1, Himg, Wimg, 3).contiguous()

        if roughness is None:
            # --- NO roughness: niente interpolazione, niente mipmapping (LOD 0)
            out = dr.texture(
                tex, wo_img,
                boundary_mode='cube',
                filter_mode='linear'  # nearest + LOD 0 implicito
            )  # [1,Himg,Wimg,C]
        else:
            # --- padding roughness (poi tagliato)
            if total != N:
                pad = total - N
                pad_r = torch.ones(pad, 1, device=device, dtype=roughness.dtype)
                roughness = torch.cat([roughness, pad_r], dim=0)

            # LOD assoluto per-pixel (roughness -> [0, max_lod])
            # NB: assumo tex.shape[2] = Hf = dimensione base per mip chain
            max_lod = torch.log2(torch.as_tensor(tex.shape[2], dtype=torch.float32, device=device))
            lod_img = (roughness.squeeze(-1).clamp(0, 1)) * max_lod
            lod_img = lod_img.view(1, Himg, Wimg)  # o (1,H,W,1) se la tua build lo preferisce

            out = dr.texture(
                tex, wo_img,
                boundary_mode='cube',
                filter_mode='linear-mipmap-linear',
                mip_level_bias=lod_img,
                max_mip_level=getattr(self, 'mip_levels', None)  # usa self.mip_levels se esiste
            )  # [1,Himg,Wimg,C]

        # --- flatten & rimuovi padding
        out = out.view(-1, C)[:N, :]

        if getattr(self, 'num_channels', C) > 3:
            out = self.mlp_decoder(out)
        return out

        
        
    def compute_specular(self, wo, pos, roughness, feat_map=None):
        
        
        if self.color_representation == 'cubemap':
            out = self.eval_envmap(wo, roughness)
            return out
        
        if self.color_representation == 'big_voronoi':
            sites_dir = F.normalize(self._sites, dim=-1)
            dist = torch.norm(sites_dir.unsqueeze(0) - wo.unsqueeze(1), dim=-1)
            K_levels = self._kappa_levels
            N = int(K_levels.shape[0])
            r = roughness.reshape(-1).clamp_min(1e-4)                     
            alpha = r                                                 
            lod = ((N - 1) * alpha).clamp(0, N - 1)                       
            l0 = lod.floor().long()                                      
            l1 = torch.clamp(l0 + 1, max=N - 1)                           
            tL = (lod - l0.float())        
            kL0 = K_levels[l0]
            kL1 = K_levels[l1]
            k_level = ((1.0 - tL) * kL0 + tL * kL1).clamp(min=1e-4)       
            W = torch.softmax(-k_level.unsqueeze(-1) * dist, dim=1).unsqueeze(-1)
            V = (W * self._colors.unsqueeze(0)).sum(dim=1)
            return V
        
        if self.color_representation == 'lp_voronoi_cmap':
            H,D,W = self.G, self.G, self.G
            grid_map = self.fill_octa_from_topk_kappa_per_voxel()
            idx8_zyx, w8 = self.world_to_voxel_trilin(pos, self.bbox_min, self.bbox_max, self.grid_res, align_corners=True)
            atlas, tiles_x, tiles_y, M_face = build_octa_atlas(grid_map)
            out = sample_with_octa_atlas(
                atlas, tiles_x, tiles_y, M_face,
                idx8_zyx, w8, wo, roughness, self.grid_res, use_r2=False
            ) 
            return out
        
        
        elif self.color_representation == 'lp_cubemaps':
            
            device = pos.device
            B = pos.shape[0]
            V = self.NUM_PROBES          # es. 128
            H = W = self.map_res         # es. 64
            C = 3
            Ffaces = V * 6               # numero facce totali
            nn_k = 8
            # Fallback se non usi probes
            if getattr(self, 'color_representation', None) != 'lp_cubemaps':
                return self.eval_envmap(wo, roughness)
            if not getattr(self, 'use_light_probes', True):
                return self.eval_envmap(wo, roughness)

            # ---------- KNN ----------
            pos_ = pos.unsqueeze(0).contiguous()              # [1,B,3]
            vox_ = self.positions.unsqueeze(0).contiguous()   # [1,V,3]
            knn = knn_points(pos_, vox_, K=nn_k, return_nn=False)
            dk = knn.dists.squeeze(0).clamp_min(0).sqrt()     # [B,K]
            iidx = knn.idx.squeeze(0).long()                  # [B,K]

            # Pesi (Wendland-like)
            h = dk[:, [-1]]
            q = dk / (h + 1e-8)
            w = (torch.clamp(1.0 - q, min=0.0) ** 4) * (4.0 * q + 1.0)
            w = torch.where(q <= 1.0, w, torch.zeros_like(w))
            weights = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B,K]
            self.chosen_probes = iidx
            self.weights = weights

            # ---------- direzioni & LOD ----------
            wo_n = F.normalize(wo, dim=-1)
            K = iidx.shape[1]
            N = B * K

            wo_expanded = wo_n.unsqueeze(1).expand(B, K, 3).reshape(N, 3)  # [N,3]
            probe_ids_flat = iidx.reshape(N).to(device)                    # [N]

            max_lod = int(math.log2(min(H, W)))
            rough = roughness.reshape(B, -1)[:, 0].clamp(0, 1)
            lod = (rough * float(max_lod)).unsqueeze(1).expand(B, K).reshape(N)  # [N]

            # ---------- dir -> (face,u,v) ----------
            face, u, v = dir_to_face_uv(wo_expanded)          # face:[N], u/v in [-1,1]
            face = face.long().to(device)


            # UV in [0,1] (flip v se ti serve)
            u = (u + 1.0) * 0.5
            v = (v + 1.0) * 0.5
            # v = 1.0 - v
            eps = 1e-6
            u = u.clamp(eps, 1.0 - eps)
            v = v.clamp(eps, 1.0 - eps)

            # indice faccia globale per sample
            img_idx = (probe_ids_flat * 6 + face).long()      # [N] in [0..Ffaces-1]

            # ---------- batch textures: tutte le facce in parallelo ----------
            # [V,6,H,W,3] -> [Ffaces,H,W,3]
            faces_tex = self.probe_cubemaps.view(Ffaces, H, W, C).contiguous()

            # ---------- impacchetta UV per faccia (no loop) ----------
            # Ordina per faccia, calcola lo "slot" per-sample all'interno della faccia
            order = torch.argsort(img_idx)                    # [N]
            img_sorted = img_idx[order]                       # [N]
            uv_sorted = torch.stack([u, v], dim=-1)[order]    # [N,2]
            lod_sorted = lod[order]                           # [N]

            # Conteggi per faccia e max punti per faccia
            counts = torch.bincount(img_sorted, minlength=Ffaces)           # [Ffaces]
            M_max = int(counts.max().item()) if counts.numel() > 0 else 0

            # Primo indice di ciascuna faccia nel vettore ordinato (usa searchsorted)
            first_idx = torch.searchsorted(img_sorted, torch.arange(Ffaces, device=device))
            # Slot per-sample = posizione - primo indice della sua faccia
            arangeN = torch.arange(N, device=device)
            slot_sorted = arangeN - first_idx[img_sorted]                    # [N], in [0 .. count-1]

            # Prepara griglie UV e bias LOD batchate e padded: [Ffaces, M_max, 1, ...]
            # Riempite con valori validi (es. 0) — useremo solo i "slot" < count
            uv_grid = torch.zeros(Ffaces, M_max, 1, 2, device=device, dtype=uv_sorted.dtype)
            lod_grid = torch.zeros(Ffaces, M_max, 1, device=device, dtype=lod_sorted.dtype)
            # scatter nei posti giusti
            uv_grid[img_sorted, slot_sorted, 0, :] = uv_sorted
            lod_grid[img_sorted, slot_sorted, 0] = lod_sorted

            # ---------- texture sample in UNA chiamata ----------
            # dr.texture supporta batch: tex [B,H,W,C], uv [B,M,1,2], bias [B,M,1]
            try:
                samp_b = dr.texture(
                    faces_tex,                     # [Ffaces, H, W, C]
                    uv_grid,                       # [Ffaces, M_max, 1, 2]
                    boundary_mode='clamp',
                    filter_mode='linear-mipmap-linear',
                    mip_level_bias=lod_grid,
                    max_mip_level=max_lod
                )                                  # -> [Ffaces, M_max, 1, C]
            except RuntimeError:
                samp_b = dr.texture(
                    faces_tex,
                    uv_grid,
                    boundary_mode='clamp',
                    filter_mode='linear'
                )                                  # -> [Ffaces, M_max, 1, C]

            # Riprendi i colori solo degli slot usati e rimetti in ordine originale
            samp_b = samp_b.squeeze(2)                              # [Ffaces, M_max, C]
            col_sorted = samp_b[img_sorted, slot_sorted]            # [N, C]
            # inverti l'ordinamento
            inv = torch.empty_like(order)
            inv[order] = torch.arange(N, device=device)
            col = col_sorted[inv]                                   # [N, C]

            # ---------- aggrega K vicini ----------
            col_BKC = col.view(B, K, C)
            out_probes = (col_BKC * weights.unsqueeze(-1)).sum(dim=1)  # [B,3]
            
            if hasattr(self, '_alpha'):
                # alpha per-probe in [0,1]
                # prendi gli alpha dei K probe selezionati e fai la media pesata
                A_q = torch.sigmoid(self._alpha.index_select(0, iidx.reshape(-1)))  # [B*K,1]
                A_q = A_q.view(B, K)                                                # [B,K]
                alpha_eff = (A_q * weights).sum(dim=1, keepdim=True)                # [B,1]

                out_env = self.eval_envmap(wo_n, roughness)                         # [B,3]
                return alpha_eff * out_env + (1.0 - alpha_eff) * out_probes

            return out_probes
        
         # lp_opt
        G = self.G
        R = self.map_res
        M = self.topk
        B = pos.shape[0]
        V, S, _ = self._sites.shape
        nn_k = self.nn_k
        idx_topk = self.idx_topk

        if self.color_representation == 'lp_opt_grid':
            
            if not self.use_light_probes:
                return self.eval_envmap(wo, roughness) 
            
            pos_ = pos.unsqueeze(0).contiguous()                     # (1,B,3)

            eps = 1e-12
            extent = (self.bbox_max - self.bbox_min).clamp_min(eps)        # (3,)
            rel = ((pos - self.bbox_min) / extent).clamp(0, 1)            # (B,3) in [0,1]^3
            p   = rel * G - 0.5                                           # (B,3) pixel-centers

            i0 = torch.floor(p).to(torch.long)                            # (B,3)
            i1 = (i0 + 1)
            i0 = i0.clamp(0, G-1); i1 = i1.clamp(0, G-1)
            t  = (p - i0.to(p.dtype)).clamp(0, 1)                         # (B,3)
            one_minus_t = 1.0 - t

            # 8 combinazioni corner via bitmask (niente where triplicati)
            B = pos.shape[0]
            offs_bits = torch.arange(8, device=pos.device, dtype=torch.long)  # 0..7
            bx = ((offs_bits >> 0) & 1).expand(B, 8)
            by = ((offs_bits >> 1) & 1).expand(B, 8)
            bz = ((offs_bits >> 2) & 1).expand(B, 8)

            idx_stack = torch.stack((i0, i1), dim=2)                    # (B,3,2)
            w_stack   = torch.stack((one_minus_t, t), dim=2)            # (B,3,2)

            ix = idx_stack[:, 0, :].gather(1, bx)                       # (B,8)
            iy = idx_stack[:, 1, :].gather(1, by)                       # (B,8)
            iz = idx_stack[:, 2, :].gather(1, bz)                       # (B,8)

            wx = w_stack[:, 0, :].gather(1, bx)                         # (B,8)
            wy = w_stack[:, 1, :].gather(1, by)                         # (B,8)
            wz = w_stack[:, 2, :].gather(1, bz)                         # (B,8)
            weights = wx * wy * wz                                           # (B,8)

            v_id = ((iz * G + iy) * G + ix).to(torch.long)              # (B,8)

            face, u, v = dir_to_face_uv(wo)                             # (B,)
            i_tex, j_tex = uv_to_ij(u, v, R)
            cell_id = face_ij_to_cellid(face, i_tex, j_tex, R)          # (B,)
            cell_b = cell_id.unsqueeze(1).expand(-1, 8)                 # (B,8)

            # 4) Candidati Top-M per ogni voxel dei 8 corner
            cand_idx = idx_topk[v_id, cell_b, :]                        # (B,8,M)

            # 5) Gather lineare su sites/colors
            V, S, _ = self._sites.shape
            sites_flat  = self._sites.reshape(V*S, 3).contiguous()
            colors_flat = self._colors.reshape(V*S, 3).contiguous()
            alphas_flat = self._alpha.reshape(V, 1).contiguous()
            lin = (v_id.unsqueeze(-1) * S + cand_idx).reshape(-1)       # (B*8*M,)

            S_q = sites_flat.index_select(0, lin).reshape(B, 8, M, 3)   # (B,8,M,3)
            C_q = colors_flat.index_select(0, lin).reshape(B, 8, M, 3)  # (B,8,M,3)
            A_q = alphas_flat.index_select(0, v_id.reshape(-1)).reshape(B, 8, 1)
            
            S_q = F.normalize(S_q, dim=-1)
            A_q = torch.sigmoid(A_q)
            
            K_levels = self._kappa_levels
            N = int(K_levels.shape[0])
            
            r = roughness.reshape(-1).clamp_min(1e-4)                     # (B,)
            alpha = r                                                 # (B,)
            lod = ((N - 1) * alpha).clamp(0, N - 1)                       # (B,)
            l0 = lod.floor().long()                                       # (B,)
            l1 = torch.clamp(l0 + 1, max=N - 1)                           # (B,)
            tL = (lod - l0.float()).unsqueeze(-1).unsqueeze(-1)           # (B,1,1)
            
            kL0 = K_levels[l0].view(B, 1, 1)
            kL1 = K_levels[l1].view(B, 1, 1)
            k_level = ((1.0 - tL) * kL0 + tL * kL1).clamp(min=1e-4)       # (B,1,1)
        
            scores = ((S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1)) * k_level   # (B,8,M)
            W = F.softmax(scores, dim=-1)                                             # (B,8,M)
            col_8 = (W.unsqueeze(-1) * C_q).sum(dim=2)                             # (B,8,3)

            out_probes = (col_8 * weights.unsqueeze(-1)).sum(dim=1)   
            
            # alphas = (A_q * weights.unsqueeze(-1)).sum(dim=1) # (B,3)
            # out_env = self.eval_envmap(wo, roughness) 

            # return (alphas * out_env) + ((1 - alphas) * out_probes)
            #return  out_probes + out_env
            return out_probes
        
        
        if self.color_representation == 'lp_opt_d':
            
            if not self.use_light_probes:
                return self.eval_envmap(wo, roughness) 
            
            pos_ = pos.unsqueeze(0).contiguous()                     # (1,B,3)
            vox_ = self.positions.unsqueeze(0).contiguous()         # (1,V,3)

            knn = knn_points(pos_, vox_, K=nn_k, return_nn=False)    # dists2: (1,B,8), idx: (1,B,8)
            dk  = knn.dists.squeeze(0).clamp_min(0).sqrt()           # (B,8)  
            iidx = knn.idx.squeeze(0)                                # (B,8)  

            alpha = 1
            h = alpha * dk[:, [-1]]                                  # (B,1)

            q = dk / (h + 1e-8)                                      # (B,8)
            w = (torch.clamp(1.0 - q, min=0.0)**4) * (4.0*q + 1.0)   # (B,8)
            w = torch.where(q <= 1.0, w, torch.zeros_like(w))        

            wsum = w.sum(dim=1, keepdim=True)
            weights = w / wsum.clamp_min(1e-8)
            
            
            if nn_k == 1:
                weights = torch.ones_like(weights)
            
            self.chosen_probes = iidx
            self.weights = weights
            
            face, u, v = dir_to_face_uv(wo)                             # (B,)
            i_tex, j_tex = uv_to_ij(u, v, R)
            cell_id = face_ij_to_cellid(face, i_tex, j_tex, R)          # (B,)
            cell_b = cell_id.unsqueeze(1).expand(-1, nn_k)                 # (B,8)
            cand_idx = idx_topk[iidx, cell_b, :]                        # (B,8,M)
            
            sites_flat  = self._sites.reshape(V*S, 3).contiguous()
            colors_flat = self._colors.reshape(V*S, self.color_dim).contiguous()
            alphas_flat = self._alpha.reshape(V, 1).contiguous()
            
            lin = (iidx.unsqueeze(-1) * S + cand_idx).reshape(-1)       # (B*8*M,)

            S_q = sites_flat.index_select(0, lin).reshape(B, nn_k, M, 3)   # (B,8,M,3)
            C_q = colors_flat.index_select(0, lin).reshape(B, nn_k, M, self.color_dim)  # (B,8,M,3)
            A_q = alphas_flat.index_select(0, iidx.reshape(-1)).reshape(B, nn_k, 1)
            
            S_q = F.normalize(S_q, dim=-1)
            A_q = torch.sigmoid(A_q)
            
            
            if self.color_dim > 3:
                C_q = C_q.reshape(C_q.shape[0], C_q.shape[1], -1)
                col_8 = self.mlp_decoder(C_q)
                col_8 = torch.sigmoid(col_8)
            else:
            
                K_levels = self._kappa_levels
                N = int(K_levels.shape[0])
                
                r = roughness.reshape(-1).clamp_min(1e-4)                     # (B,)
                alpha = r                                                 # (B,)
                lod = ((N - 1) * alpha).clamp(0, N - 1)                       # (B,)
                l0 = lod.floor().long()                                       # (B,)
                l1 = torch.clamp(l0 + 1, max=N - 1)                           # (B,)
                tL = (lod - l0.float()).unsqueeze(-1).unsqueeze(-1)           # (B,1,1)
                
                kL0 = K_levels[l0].view(B, 1, 1)
                kL1 = K_levels[l1].view(B, 1, 1)
                k_level = ((1.0 - tL) * kL0 + tL * kL1).clamp(min=1e-4)       # (B,1,1)
                
                #scores = ((S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1)) * k_level   # (B,8,M)
                
                scores = torch.norm((S_q - wo.unsqueeze(1).unsqueeze(1)), dim=-1) * k_level   # (B,8,M)
                W = F.softmax(scores, dim=-1)                                             # (B,8,M)
                col_8 = (W.unsqueeze(-1) * C_q).sum(dim=2)                             # (B,8,3)

            out_probes = (col_8 * weights.unsqueeze(-1)).sum(dim=1)   
            alphas = (A_q * weights.unsqueeze(-1)).sum(dim=1) # (B,3)
            
            out_env = self.eval_envmap(wo, roughness) 

            return (alphas * out_env) + ((1 - alphas) * out_probes)
            #return  out_probes + out_env
            # return out_probes
            
            
        if self.use_mlp:
            # enc_p = positional_encoding(self.voxel_centers.reshape(-1, 3), self.num_pos_freq)
            # inp = torch.cat([self.features, enc_p], dim=-1)
            # sites = self.mlp_sites(self.features).reshape(-1, self.K, 3)
            sites = self._pred_sites
            colors = self.mlp_color(self.features).reshape(-1, self.K, 3)
        else:
            colors = self._colors   
            sites = self._sites

        
        eps = 1e-6
        extent = (self.bbox_max - self.bbox_min).clamp_min(eps)        
        rel = ((pos - self.bbox_min) / extent).clamp(0, 1)            
        p   = rel * G - 0.5                                           
        i0 = torch.floor(p).to(torch.long)                            
        i1 = (i0 + 1)
        i0 = i0.clamp(0, G-1); i1 = i1.clamp(0, G-1)
        t  = (p - i0.to(p.dtype)).clamp(0, 1)                         
        one_minus_t = 1.0 - t
        
        B = pos.shape[0]
        offs_bits = torch.arange(8, device=pos.device, dtype=torch.long)  
        bx = ((offs_bits >> 0) & 1).expand(B, 8)
        by = ((offs_bits >> 1) & 1).expand(B, 8)
        bz = ((offs_bits >> 2) & 1).expand(B, 8)
        
        idx_stack = torch.stack((i0, i1), dim=2)                    
        w_stack   = torch.stack((one_minus_t, t), dim=2)            

        ix = idx_stack[:, 0, :].gather(1, bx)                       
        iy = idx_stack[:, 1, :].gather(1, by)                       
        iz = idx_stack[:, 2, :].gather(1, bz)                       

        wx = w_stack[:, 0, :].gather(1, bx)                        
        wy = w_stack[:, 1, :].gather(1, by)                         
        wz = w_stack[:, 2, :].gather(1, bz)                        
        w8 = wx * wy * wz                                           
        v_id = ((iz * G + iy) * G + ix).to(torch.int32)
        face, u, v = dir_to_face_uv(wo) 
        i_tex, j_tex = uv_to_ij(u, v, R)
        cell_id = face_ij_to_cellid(face, i_tex, j_tex, R) 
        cell_b = cell_id.unsqueeze(1).expand(-1, 8)
        
        cand_idx = idx_topk[v_id, cell_b, :]
        V, S, _ = sites.shape
        sites_flat  = sites.reshape(V*S, 3).contiguous()
        colors_flat = colors.reshape(V*S, self.color_dim).contiguous()
        alphas_flat = self._alpha.reshape(V, 1).contiguous()
        lin = (v_id.unsqueeze(-1) * S + cand_idx).reshape(-1)
        
        S_q = sites_flat.index_select(0, lin).reshape(B, 8, M, 3)   # (B,8,M,3)
        C_q = colors_flat.index_select(0, lin).reshape(B, 8, M, self.color_dim)  # (B,8,M,3)
        #K_q = kappas_flat.index_select(0, lin).reshape(B, 8, M, 1)  # (B,8,M,3)
        A_q = alphas_flat.index_select(0, v_id.reshape(-1)).reshape(B, 8, 1, 1)  # (B,8,M,3)
        #K_q = F.softplus(K_q)
        S_q = F.normalize(S_q, dim=-1)
        A_q = torch.sigmoid(A_q)
        K_levels = self._kappa_levels
        N = int(K_levels.shape[0])
        
        r = roughness.reshape(-1).clamp_min(1e-4)                     # (B,)
        alpha = r                                                 # (B,)
        lod = ((N - 1) * alpha).clamp(0, N - 1)                       # (B,)
        l0 = lod.floor().long()                                       # (B,)
        l1 = torch.clamp(l0 + 1, max=N - 1)                           # (B,)
        tL = (lod - l0.float()).unsqueeze(-1).unsqueeze(-1)           # (B,1,1)


        kL0 = K_levels[l0].view(B, 1, 1)
        kL1 = K_levels[l1].view(B, 1, 1)
        k_level = ((1.0 - tL) * kL0 + tL * kL1).clamp(min=1e-4)       # (B,1,1)
       
        #cos = (S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1).clamp_min(0.0)
        scores = ((S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1)) * k_level   # (B,8,M)
        #scores = k_level * (cos - 1)
        #scores = (-K_q.squeeze(-1) * (1 - (S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1))) / (1 + roughness.unsqueeze(-1))
        W = F.softmax(scores, dim=-1)                                             # (B,8,M)
        col_8 = (W.unsqueeze(-1) * C_q).sum(dim=2)                             # (B,8,3)

        out_colors = (col_8 * w8.unsqueeze(-1)).sum(dim=1)                     # (B,3)

        if self.color_dim > 3:
            out_colors = self.mlp_decoder(out_colors)
        return out_colors 
                    
    
    def compute_specular_lod(self, wo, pos, roughness, feat_map=None):

        G = self.G
        R = self.map_res
        M = self.topk

        sites = F.normalize(self._sites, dim=-1)
        colors = self._colors
        kappas = self._kappas          # κ base per (voxel,site)
        idx_topk = self.idx_topk

        eps = 1e-6
        extent = (self.bbox_max - self.bbox_min).clamp_min(eps)
        rel = ((pos - self.bbox_min) / extent).clamp(0, 1)
        p = rel * G - 0.5
        i0 = torch.floor(p).to(torch.long); i1 = (i0 + 1)
        i0 = i0.clamp(0, G-1); i1 = i1.clamp(0, G-1)
        t = (p - i0.to(p.dtype)).clamp(0, 1); one_minus_t = 1.0 - t

        B = pos.shape[0]
        offs_bits = torch.arange(8, device=pos.device, dtype=torch.long)
        bx = ((offs_bits >> 0) & 1).expand(B, 8)
        by = ((offs_bits >> 1) & 1).expand(B, 8)
        bz = ((offs_bits >> 2) & 1).expand(B, 8)

        idx_stack = torch.stack((i0, i1), dim=2)
        w_stack   = torch.stack((one_minus_t, t), dim=2)

        ix = idx_stack[:, 0, :].gather(1, bx)
        iy = idx_stack[:, 1, :].gather(1, by)
        iz = idx_stack[:, 2, :].gather(1, bz)

        wx = w_stack[:, 0, :].gather(1, bx)
        wy = w_stack[:, 1, :].gather(1, by)
        wz = w_stack[:, 2, :].gather(1, bz)
        w8 = wx * wy * wz
        v_id = ((iz * G + iy) * G + ix).to(torch.int32)

        face, u, v = dir_to_face_uv(wo)
        i_tex, j_tex = uv_to_ij(u, v, R)
        cell_id = face_ij_to_cellid(face, i_tex, j_tex, R)
        cell_b = cell_id.unsqueeze(1).expand(-1, 8)

        cand_idx = idx_topk[v_id, cell_b, :]                          # (B,8,M)
        V, S, _ = sites.shape
        sites_flat  = sites.reshape(V*S, 3).contiguous()
        colors_flat = colors.reshape(V*S, 3).contiguous()
        kappas_flat = kappas.reshape(V*S, 1).contiguous()
        lin = (v_id.unsqueeze(-1) * S + cand_idx).reshape(-1)

        S_q = sites_flat.index_select(0, lin).reshape(B, 8, M, 3)     # (B,8,M,3)
        C_q = colors_flat.index_select(0, lin).reshape(B, 8, M, 3)    # (B,8,M,3)
        #K_q = kappas_flat.index_select(0, lin).reshape(B, 8, M, 1)    # (B,8,M,1)


        K_levels = self._kappa_levels

        N = int(K_levels.shape[0])

        # roughness per-pixel -> alpha percettiva -> LOD continuo in [0, N-1]
        r = roughness.reshape(-1).clamp_min(1e-4)                     # (B,)
        alpha = r**2                                           # (B,)
        lod = ((N - 1) * alpha).clamp(0, N - 1)                       # (B,)
        l0 = lod.floor().long()                                       # (B,)
        l1 = torch.clamp(l0 + 1, max=N - 1)                           # (B,)
        tL = (lod - l0.float()).unsqueeze(-1).unsqueeze(-1)           # (B,1,1)

        
        kL0 = K_levels[l0].view(B, 1, 1)
        kL1 = K_levels[l1].view(B, 1, 1)
        k_level = ((1.0 - tL) * kL0 + tL * kL1).clamp(min=1e-4)       # (B,1,1)

        # === scoring e softmax (stabile, per-pixel) ===================================
        cos_th = (S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1).clamp(-1, 1)  # (B,8,M)
        logits = k_level * cos_th                            # (B,8,M)

        W = F.softmax(logits, dim=-1)                                 

        col_8 = (W.unsqueeze(-1) * C_q).sum(dim=2)                     
        out_colors = (col_8 * w8.unsqueeze(-1)).sum(dim=1)             # (B,3)
        return out_colors
        
        
    def compute_specular2(self, wo, pos, roughness, feature_map=None):
        
        
        if self.color_representation == 'big_voronoi':
            sites_dir = F.normalize(self._sites, dim=-1)
            dist = torch.norm(sites_dir.unsqueeze(0) - wo.unsqueeze(1), dim=-1)
            betas = torch.norm(self._sites, dim=-1)
            W = torch.softmax(-betas.unsqueeze(0) * dist, dim=1).unsqueeze(-1)
            V = (W * self._colors.unsqueeze(0)).sum(dim=1)
            return V
        else:
            V, K = pos.shape[0], self.max_degree

            rel   = (pos - self.bbox_min) / (self.bbox_max - self.bbox_min)
            coords = rel * 2 - 1                              # (V, 3), xyz

            grid = coords.view(1, 1, 1, V, 3)                 # (1, 1, 1, V, 3) xyz

            if self.use_mlp:
                voxel_coords = self.voxel_centers.reshape(-1, 3)
                pos = positional_encoding(voxel_coords, num_freqs=self.num_pos_freq)
                
                feat = (self._features.permute(0, 2, 3, 4, 1).reshape(-1, self.feat_dim))  # (R^3, C)
                #inp = self._features.reshape(-1, self.feat_dim)
                inp = torch.cat([feat, pos], dim=-1)
                sites = self.mlp_sites(inp).view(1, self.res, self.res, self.res, 3*self.max_degree).permute(0, 4, 1, 2, 3).contiguous()
                colors = self.mlp_color(inp).view(1, self.res, self.res, self.res, 3*self.max_degree).permute(0, 4, 1, 2, 3).contiguous()

            else:
                sites = self._sites
                colors = self._colors
                
            sites = F.grid_sample(sites, grid, mode='bilinear', align_corners=True)
            colors = F.grid_sample(colors, grid, mode='bilinear', align_corners=True)
        
            sites = sites.squeeze(0).squeeze(-1).squeeze(-1).reshape(self.max_degree, 3, -1)
            colors = colors.squeeze(0).squeeze(-1).squeeze(-1).reshape(self.max_degree, 3, -1)
            
            betas = torch.norm(sites, dim=1)
            sites_dir = F.normalize(sites, dim=1)
            
            self.betas = betas 
            
            dist = 1 - torch.sum(sites_dir * wo.permute(1, 0).unsqueeze(0), dim=1)
            logit = (-betas * dist)
            W = torch.softmax(logit, dim=0)
            self.W = W
            c_spec = (W.unsqueeze(1) * colors).sum(dim=0).permute(1, 0)
            return c_spec
                #dist = torch.norm(sites_dir - wo.unsqueeze(1), dim=-1)
                

        # top_dots, top_idx = torch.topk(dots, k=Kp, dim=1)

        # colors_k = colors_out.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
        # sites_mag = torch.norm(sites_out, dim=-1)
        # betas_k = sites_mag.gather(1, top_idx) / (1 + roughness)

        # logits_k = betas_k * top_dots - betas_k

        # m = logits_k.max(dim=1, keepdim=True).values
        # exp_shifted = torch.exp(logits_k - m)
        # den = exp_shifted.sum(dim=1, keepdim=True)
        # Wk = (exp_shifted / den).unsqueeze(-1)

        # c_spec = (Wk * colors_k).sum(dim=1)
        # self.W = Wk
 
    

    def shade_topk(self, directions, k_keep: int=64):
        V, K = self._xyz.shape[0], self.max_degree
        D, H, W = self._colors.shape[2], self._colors.shape[1], self._colors.shape[0]

        # --- build volumes (per call, no caching) ---
        if self.use_mlp:
            inp = self._features.reshape(-1, self.feat_dim)             # (res^3, feat_dim)
            sites  = self.mlp_sites(inp).view(self.res, self.res, self.res, K, 3)
            colors = self.mlp_color(inp).view(self.res, self.res, self.res, K, 3)
        else:
            sites  = self._sites     # (W,H,D,K,3)
            colors = self._colors    # (W,H,D,K,3)

        # (W,H,D,K,3) -> (C,D,H,W) with C = K*3
        colors = colors.movedim((0,1,2,3,4),(2,1,0,3,4)).reshape(D, H, W, -1).movedim(-1, 0).contiguous()
        sites  =  sites.movedim((0,1,2,3,4),(2,1,0,3,4)).reshape(D, H, W, -1).movedim(-1, 0).contiguous()

        # --- reflection dir (cheaper normalize) ---
        normals = self.get_normals()  # (V,3)
        dn = (directions * normals).sum(-1, keepdim=True)
        reflective_dir = directions - 2.0 * dn * normals
        eps = 1e-12
        inv = torch.rsqrt(torch.clamp((reflective_dir * reflective_dir).sum(-1, keepdim=True), min=eps))
        q = reflective_dir * inv  # (V,3)

        # --- sampling grid (per call, no caching) ---
        rel = (self.get_xyz - self.bbox_min) / (self.bbox_max - self.bbox_min)  # (V,3) in [0,1]
        coords = rel * 2.0 - 1.0                                                # [-1,1]
        grid = coords[:, [2, 1, 0]].view(1, V, 1, 1, 3).contiguous()            # (1,V,1,1,3)

        # --- single grid_sample for both volumes ---
        both = torch.cat([colors, sites], dim=0).unsqueeze(0)  # (1, Ctot, D,H,W)
        sampled = F.grid_sample(both, grid, mode='bilinear', align_corners=True)  # (1,Ctot,V,1,1)

        Cc = colors.shape[0]
        sc = sampled[:, :Cc]     # (1,3K,V,1,1)
        ss = sampled[:, Cc:]     # (1,3K,V,1,1)

        # -> (V,K,3)
        colors_out = sc.squeeze(0).squeeze(-1).squeeze(-1).movedim(0, 1).view(V, K, 3).contiguous()
        sites_out  = ss.squeeze(0).squeeze(-1).squeeze(-1).movedim(0, 1).view(V, K, 3).contiguous()

        # --- specular via SDPA (TOP-K + bias-folded, mask-free) ---
        # betas = ||site|| ; sites_dir = normalize(site)
        betas = torch.linalg.norm(sites_out, dim=-1, keepdim=True)  # (V,K,1)
        inv = torch.rsqrt(torch.clamp((sites_out * sites_out).sum(-1, keepdim=True), min=eps))
        sites_dir = sites_out * inv                                 # (V,K,3)

        # scale = betas / softplus(roughness)
        scale = betas.squeeze(-1) / F.softplus(self._roughness)     # (V,K)

        # ---- TOP-K by the true logits: scale * (q·mu - 1) ----
        qmu = (sites_dir * q.view(V,1,3)).sum(-1)                   # (V,K)
        scores = scale * (qmu - 1.0)                                # (V,K)

        k_keep = min(int(k_keep), K)
        if k_keep < K:
            topk_vals, topk_idx = torch.topk(scores, k_keep, dim=1)                 # (V,k)
            idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, 3)                          # (V,k,3)
            sd = torch.gather(sites_dir, 1, idx3)                                    # (V,k,3)
            sc_top = torch.gather(scale,     1, topk_idx)                            # (V,k)
            cv = torch.gather(colors_out,1, idx3)                                    # (V,k,3)
        else:
            sd, sc_top, cv = sites_dir, scale, colors_out
            k_keep = K

        # ---- Flash-compatible SDPA (fold -scale into an extra dim; NO attn_mask) ----
        d = 3
        d_prime = d + 1
        sqrt_dprime = d_prime ** 0.5

        # Q': append a constant component sqrt(d')
        Q = q.view(V, 1, 1, d)                                                       # (V,1,1,3)
        Qp = torch.cat([Q, torch.ones_like(Q[..., :1]) * sqrt_dprime], dim=-1)       # (V,1,1,4)

        # K': first d comps = (mu * scale) * sqrt(d'), extra comp = (-scale)
        K_main  = (sd * sc_top.unsqueeze(-1)) * sqrt_dprime                          # (V,k,3)
        K_extra = (-sc_top).unsqueeze(-1)                                            # (V,k,1)
        Kp = torch.cat([K_main, K_extra], dim=-1).view(V, 1, k_keep, d_prime)        # (V,1,k,4)

        Vvals = cv.view(V, 1, k_keep, 3)                                             # (V,1,k,3)

        # No attn_mask, so PyTorch can pick Flash/MemEff backends
        c_spec = torch.nn.functional.scaled_dot_product_attention(
            Qp, Kp, Vvals, attn_mask=None, is_causal=False
        ).view(V, 3)

        return torch.clamp_min(self.get_albedo + c_spec, 0.0)
    
    
    def shade(self, directions):
        V, K = self._xyz.shape[0], self.max_degree
        D, H, W = self._colors.shape[2], self._colors.shape[1], self._colors.shape[0]

        if self.use_mlp:
            # (D*H*W, feat_dim) -> heads -> (C,D,H,W) directly (no 5D movedim)
            inp = self._features.reshape(-1, self.feat_dim)                  # (D*H*W, feat_dim)
            sites_flat  = self.mlp_sites(inp)                                # (D*H*W, 3K)
            colors_flat = self.mlp_color(inp)                                # (D*H*W, 3K)

            # -> (C,D,H,W)
            colors = colors_flat.view(D, H, W, 3*K).permute(3, 0, 1, 2).contiguous()   # (3K,D,H,W)
            sites  =  sites_flat.view(D, H, W, 3*K).permute(3, 0, 1, 2).contiguous()   # (3K,D,H,W)
        else:
            # Convert once to (C,D,H,W) and reuse (no per-call movedim)
            if not hasattr(self, "_colors_cf"):
                def _whdk3_to_cdhw(x):  # (W,H,D,K,3) -> (3K,D,H,W)
                    D_, H_, W_ = x.shape[2], x.shape[1], x.shape[0]
                    return (x.permute(2,1,0,3,4)       # (D,H,W,K,3)
                            .reshape(D_, H_, W_, -1)  # (D,H,W,3K)
                            .permute(3,0,1,2)         # (3K,D,H,W)
                            .contiguous())
                # cache as buffers/params (match mutability of originals)
                self._colors_cf = torch.nn.Parameter(_whdk3_to_cdhw(self._colors))
                self._sites_cf  = torch.nn.Parameter(_whdk3_to_cdhw(self._sites))
            colors = self._colors_cf
            sites  = self._sites_cf

        # --- reflection dir (cheaper normalize) ---
        normals = self.get_normals()  # (V,3)
        dn = (directions * normals).sum(-1, keepdim=True)
        reflective_dir = directions - 2.0 * dn * normals
        eps = 1e-12
        inv = torch.rsqrt(torch.clamp((reflective_dir * reflective_dir).sum(-1, keepdim=True), min=eps))
        reflective_dir = reflective_dir * inv  # (V,3)

        # --- sampling grid (per call, no caching) ---
        rel = (self.get_xyz - self.bbox_min) / (self.bbox_max - self.bbox_min)  # (V,3) in [0,1]
        coords = rel * 2.0 - 1.0                                                # [-1,1]
        grid = coords[:, [2, 1, 0]].view(1, V, 1, 1, 3).contiguous()            # (1,V,1,1,3)

        # --- single grid_sample for both volumes ---
        both = torch.cat([colors, sites], dim=0).unsqueeze(0)  # (1, Ctot, D,H,W)
        sampled = F.grid_sample(both, grid, mode='bilinear', align_corners=True)  # (1,Ctot,V,1,1)

        # --- cheap reshapes to (V,K,3) (no squeeze/movedim chains) ---
        sc = sampled[:, :3*K, :, 0, 0].squeeze(0)                                     # (3K, V)
        ss = sampled[:, 3*K:, :, 0, 0].squeeze(0)                                     # (3K, V)

        colors_out = sc.view(3, K, V).permute(2, 1, 0).contiguous()                   # (V, K, 3)
        sites_out  = ss.view(3, K, V).permute(2, 1, 0).contiguous()   

        # --- specular via SDPA (fused softmax @ values) ---
        # betas = ||site|| ; sites_dir = normalize(site)
        betas = torch.linalg.norm(sites_out, dim=-1, keepdim=True)  # (V,K,1)
        inv = torch.rsqrt(torch.clamp((sites_out * sites_out).sum(-1, keepdim=True), min=eps))
        sites_dir = sites_out * inv                                  # (V,K,3)

        # scale = betas / softplus(roughness)
        scale = betas.squeeze(-1) / self.get_roughness         # (V,K)
        d = 3

        Q = reflective_dir.view(V, 1, 1, d)                              # (V,1,1,3)  <-- no sqrt(d)
        K_ = (sites_dir * scale.unsqueeze(-1)) * (d ** 0.5)              # (V,K,3)    <-- only K gets sqrt(d)
        K_ = K_.view(V, 1, K, d)
        Vvals = colors_out.view(V, 1, K, 3)

        mask = (-scale).view(V, 1, 1, K).to(Q.dtype)                     # additive bias per key

        c_spec = torch.nn.functional.scaled_dot_product_attention(
            Q, K_, Vvals,
            attn_mask=mask,
            is_causal=False
        ).view(V, 3)

        return torch.clamp_min(self.get_albedo + c_spec, 0.0)
            
    
    def get_color(self, directions):
        
        if self.color_representation == 'sgs':
            if self.active_degree == 0:
                return torch.clamp_min(self._dc, 0.0)
            mu = self._mu[:, 0:self.active_degree, :]
            directions = directions.reshape(-1, 1, 3)
            dot_product = (mu * directions).sum(dim=-1).unsqueeze(-1)
            gaussian_values = (self._alpha[:, 0:self.active_degree, :]) * torch.exp(dot_product - torch.norm(mu, dim=-1, keepdim=True))
            return self._dc + gaussian_values.sum(dim=1)
        elif self.color_representation == 'voronoi':
            directions = directions.reshape(-1, 1, 3)
            sites_dir = F.normalize(self._sites, dim=-1)
            dist = torch.norm(sites_dir - directions, dim=-1)
            betas = self._sites.norm(dim=-1)
            W = torch.softmax(-betas * dist, dim=-1).unsqueeze(-1)
            W = W * self._mask  
            W = W / (W.sum(dim=1, keepdim=True) + 1e-6)  
            V = (W * self._lambd).sum(dim=1)
            return torch.clamp_min(V, 0.0)
        
        elif self.color_representation == 'voronoi_r':
            normals = self.get_normals()
            wo = F.normalize(directions - 2 * (directions * normals).sum(dim=-1, keepdim=True) * normals, dim=-1)
            sites_dir = F.normalize(self._sites, dim=-1)
            dist = 1 - (wo.unsqueeze(1) * sites_dir).sum(dim=-1)
            betas = self._sites.norm(dim=-1)
            logit = (-betas * dist) / (1 + self.get_roughness**2)
            W = torch.softmax(logit, dim=-1).unsqueeze(-1)
            V = (W * self._lambd).sum(dim=1)
            return self._dc + V
        
        elif self.color_representation == 'lp_opt':
            normals = self.get_normals()
            wo = F.normalize(directions - 2 * (directions * normals).sum(dim=-1, keepdim=True) * normals, dim=-1)
            V = self.compute_specular(wo, self._xyz, self.get_roughness)
            return self._dc + V
            
        elif self.color_representation == 'light_probes2':
            D, H, W = self._colors.shape[2], self._colors.shape[1], self._colors.shape[0]
            V, K = self._xyz.shape[0], self.max_degree
            normals = self.get_normals()
            reflective_dir = F.normalize(directions - 2 * (directions * normals).sum(dim=-1, keepdim=True) * normals, dim=-1)
            rel = (self.get_xyz - self.bbox_min) / (self.bbox_max - self.bbox_min)
            coords = rel * 2 - 1
            grid = coords[:, [2, 1, 0]].view(1, self._xyz.shape[0], 1, 1, 3)
            if self.use_mlp:
                voxel_coords = self.voxel_centers.reshape(-1, 3)  # (res³, 3)
                inp = self._features.reshape(-1, self.feat_dim)
                sites_flat = self.mlp_sites(inp)
                colors_flat = self.mlp_color(inp)
                colors = colors_flat.view(D, H, W, 3*K).permute(3, 0, 1, 2).contiguous()   # (3K,D,H,W)
                sites  =  sites_flat.view(D, H, W, 3*K).permute(3, 0, 1, 2).contiguous()   # (3K,D,H,W)
            else:
                sites = self._sites
                colors = self._colors

            both = torch.cat([colors, sites], dim=0).unsqueeze(0)  # (1, Ctot, D,H,W)
            sampled = F.grid_sample(both, grid, mode='bilinear', align_corners=True)  # (1,Ctot,V,1,1)

            sc = sampled[:, :3*K, :, 0, 0].squeeze(0)                                     # (3K, V)
            ss = sampled[:, 3*K:, :, 0, 0].squeeze(0)                                     # (3K, V)

            colors_out = sc.view(3, K, V).permute(2, 1, 0).contiguous()                   # (V, K, 3)
            sites_out  = ss.view(3, K, V).permute(2, 1, 0).contiguous()   
            
            V, K = colors_out.shape[0], colors_out.shape[1]
            Kp = min(self.topk, K)

            sites_dir = F.normalize(sites_out, dim=-1)
            dots = torch.bmm(sites_dir, reflective_dir.unsqueeze(-1)).squeeze(-1)

            top_dots, top_idx = torch.topk(dots, k=Kp, dim=1)

            colors_k = colors_out.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
            sites_mag = torch.norm(sites_out, dim=-1)
            betas_k = sites_mag.gather(1, top_idx) / self.get_roughness

            logits_k = betas_k * top_dots - betas_k

            m = logits_k.max(dim=1, keepdim=True).values
            exp_shifted = torch.exp(logits_k - m)
            den = exp_shifted.sum(dim=1, keepdim=True)
            Wk = (exp_shifted / den).unsqueeze(-1)

            c_spec = (Wk * colors_k).sum(dim=1)

            self.W = Wk
            out = self.get_albedo + c_spec
            return out
            
            
        elif self.color_representation == 'light_probes':
            if self.use_light_probes:
                D, H, W = self._colors.shape[2], self._colors.shape[1], self._colors.shape[0]
                V, K = self._xyz.shape[0], self.max_degree
                normals = self.get_normals()
                reflective_dir = F.normalize(directions - 2 * (directions * normals).sum(dim=-1, keepdim=True) * normals, dim=-1)
                rel = (self.get_xyz - self.bbox_min) / (self.bbox_max - self.bbox_min)
                coords = rel * 2 - 1
                grid = coords[:, [2, 1, 0]].view(1, self._xyz.shape[0], 1, 1, 3)
                
                if self.use_mlp:
                    voxel_coords = self.voxel_centers.reshape(-1, 3)  # (res³, 3)
                    #inp = positional_encoding(voxel_coords, self.num_pos_freq)
                    inp = self._features.reshape(-1, self.feat_dim)
                    #inp = torch.cat([inp, self._features], dim=-1)
                    sites_flat = self.mlp_sites(inp)
                    colors_flat = self.mlp_color(inp)
                    
                    colors = colors_flat.view(D, H, W, 3*K).permute(3, 0, 1, 2).contiguous()   # (3K,D,H,W)
                    sites  =  sites_flat.view(D, H, W, 3*K).permute(3, 0, 1, 2).contiguous()   # (3K,D,H,W)
                    
                    # inp = sites.reshape(-1, 3)
                    # inp = positional_encoding(inp, self.num_pos_freq).reshape(self.res**3, self.max_degree, -1)
                    # logits = self.gating_mlp(inp).squeeze(-1)
                    # probs = F.softmax(logits, dim=-1)
                    # topk_vals, topk_idx = torch.topk(probs, k=self.topk, dim=-1)
                    # threshold = topk_vals[:, -1].unsqueeze(-1)               # (N, 1)
                    # hard_mask = (probs >= threshold).float()                 # (N, K)
                    # mask = (hard_mask + probs - probs.detach()).reshape(self.res, self.res, self.res, self.max_degree, -1)
                    # colors = colors * mask
                    # self.prob_mask = mask
                else:
                    sites = self._sites
                    colors = self._colors
                
                both = torch.cat([colors, sites], dim=0).unsqueeze(0)  # (1, Ctot, D,H,W)
                sampled = F.grid_sample(both, grid, mode='bilinear', align_corners=True)  # (1,Ctot,V,1,1)

                sc = sampled[:, :3*K, :, 0, 0].squeeze(0)                                     # (3K, V)
                ss = sampled[:, 3*K:, :, 0, 0].squeeze(0)                                     # (3K, V)

                colors_out = sc.view(3, K, V).permute(2, 1, 0).contiguous()                   # (V, K, 3)
                sites_out  = ss.view(3, K, V).permute(2, 1, 0).contiguous()   
                #betas_out = F.grid_sample(betas, grid, mode='bilinear', align_corners=True)
                    
        
                # betas  = betas_out.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0).view(V, K, 1)
                # betas = F.softplus(betas)
                betas = torch.norm(sites_out, dim=-1, keepdim=True)        # (V, K, 1)
                
                # if self.warm_up:
                #     betas = betas.clamp_max(5)
                # else:
                #     betas = betas.clamp_max(100)
                
                sites_dir = F.normalize(sites_out, dim=-1)                        # (V, K, 3)
                #dist = torch.norm(sites_dir - reflective_dir.unsqueeze(1), dim=-1)         # (V, K)
                dist = 1 - torch.sum(sites_dir * reflective_dir.unsqueeze(1), dim=-1)
                logit = (-betas.squeeze(-1) * dist) /  (1+self.get_roughness)
                #logit = -F.softplus(self._roughness) * dist
                W = torch.softmax(logit, dim=1).unsqueeze(-1)
                c_spec = (W * colors_out).sum(dim=1)
                self.W = W
                #c_spec = torch.exp(torch.clamp(c_spec, max=5.0))
                return self.get_albedo + c_spec
            else:
                return self.get_albedo
        
    @torch.no_grad() 
    def _create_probe_atlas(self):
        """
        Crea un atlas statico con TUTTE le probe impacchettate in una griglia.
        Chiamato UNA VOLTA all'init.
        """
        V = self.NUM_PROBES  # 128
        H = W = self.map_res  # 64
        
        # Layout: tutte le 768 facce (128 probe * 6 faces) in una griglia
        total_faces = V * 6  # 768
        
        # Griglia quasi-quadrata
        faces_per_row = int(math.ceil(math.sqrt(total_faces)))  # ~28
        num_rows = int(math.ceil(total_faces / faces_per_row))
        
        atlas_h = num_rows * H
        atlas_w = faces_per_row * W
        
        # Crea atlas vuoto
        atlas = torch.zeros(1, atlas_h, atlas_w, 3, device='cuda', dtype=torch.float32)
        
        # Mapping table: [V, 6, 4] con (u_offset, v_offset, u_scale, v_scale) in [0,1]
        probe_face_mapping = torch.zeros(V, 6, 4, device='cuda', dtype=torch.float32)
        
        face_idx = 0
        for probe_id in range(V):
            for face_id in range(6):
                row = face_idx // faces_per_row
                col = face_idx % faces_per_row
                
                y_start = row * H
                x_start = col * W
                
                # Copia la faccia nell'atlas
                atlas[0, y_start:y_start+H, x_start:x_start+W, :] = self.probe_cubemaps[probe_id, face_id]
                
                # Salva mapping in [0,1]
                u_offset = x_start / atlas_w
                v_offset = y_start / atlas_h
                u_scale = W / atlas_w
                v_scale = H / atlas_h
                
                probe_face_mapping[probe_id, face_id] = torch.tensor([u_offset, v_offset, u_scale, v_scale])
                
                face_idx += 1
        
        # Salva come buffer (non ottimizzabile) - le cubemap originali sono già Parameter
        self.probe_atlas = atlas
        self.probe_face_mapping = probe_face_mapping
        
        print(f"✓ Created probe atlas: {atlas_h}x{atlas_w} ({atlas.numel()*4/1e6:.1f} MB)")


    def update_sites_mask(self):
        self._mask = torch.ones_like(self._lambd)
        m = self._lambd.abs().sum(dim=-1) < 1e-3
        self._mask[m] = 0

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)


    def get_normals(self):
        R = build_rotation(self._rotation) 
        normals = R[:, :, 2]                
        return normals


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.color_representation == 'sgs':
            mu = fibonacci_sphere(self.max_degree) 
            mu = mu.repeat(self._xyz.shape[0], 1, 1).to('cuda')
            self._mu = nn.Parameter(mu)
            self._alpha = nn.Parameter(torch.ones((self._xyz.shape[0], self.max_degree, 3), device='cuda', requires_grad=True) * 0.05)
            init_lambda = 8
            raw_lambda_init = torch.log(torch.expm1(torch.tensor(init_lambda, device='cuda')))
            self._lambda = nn.Parameter(
                raw_lambda_init.expand(self._xyz.shape[0], self.max_degree, 1).clone()
            )
            self._dc = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), device='cuda', requires_grad=True))
            
        elif self.color_representation == 'voronoi':
            
            init_beta = 1
            vectors = fibonacci_sphere(self.max_degree)
            vectors = vectors.repeat(self._xyz.shape[0], 1, 1).to('cuda') #* 0.01

            self._sites = nn.Parameter(vectors)
            self._beta = nn.Parameter(torch.ones((self._sites.shape[0], self.active_degree, 1), device='cuda') * init_beta)     
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            self._lambd  = nn.Parameter(self._dc.detach().repeat(1, self.max_degree).reshape(self._xyz.shape[0], -1, 3)) 
            self._mask = torch.ones_like(self._lambd)
            
            
        elif self.color_representation == 'big_voronoi':
            self.big_n = 1024
            self._kappa_levels = torch.tensor([1500, 900, 320, 110, 38.4, 13.4, 4.7, 1.64, 0.57, 0.20], device='cuda', dtype=torch.float32)
            vectors = fibonacci_sphere(self.big_n).to('cuda').reshape(-1, 3) * 5
            self._sites = nn.Parameter(vectors) 
            self._colors = nn.Parameter(torch.ones((self.big_n, 3), device='cuda') * 0.5)
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0)
            self._roughness = nn.Parameter(torch.full( (self._dc.shape[0], 1), fill_value=0.0, device='cuda'))
            
        elif self.color_representation == 'cubemap':
            hidden_dim = 128
            num_hidden_layers = 2
            self.bbox_min = self._xyz.min(dim=0)[0]
            self.bbox_max = self._xyz.max(dim=0)[0]
            #self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0 + 0.25)
            self._roughness = nn.Parameter(torch.full( (self._dc.shape[0], 1), fill_value=0.0, device='cuda'))
            self.mlp_normals = nn.Sequential(*(
                [nn.Linear(self.num_pos_freq *2*3 +3, hidden_dim), nn.ReLU(True)]
                + sum([[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
                    for _ in range(num_hidden_layers-1)], [])
                + [nn.Linear(hidden_dim, 3)]
            )).to('cuda')
            
        elif self.color_representation == 'voronoi_r':
            init_beta = 1
            vectors = fibonacci_sphere(self.max_degree)
            vectors = vectors.repeat(self._xyz.shape[0], 1, 1).to('cuda') #* 0.01
            self._sites = nn.Parameter(vectors)
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            self._lambd  = nn.Parameter(self._dc.detach().repeat(1, self.max_degree).reshape(self._xyz.shape[0], -1, 3)) 
            self._roughness = nn.Parameter(torch.full( (self._dc.shape[0], 1), fill_value=0.0, device='cuda'))
        
        elif self.color_representation == 'lp_voronoi_cmap':
            init_kappa = 256
            self.color_dim = 4
            hidden_dim = 128
            num_hidden_layers = 2
            self.grid_res = torch.tensor([self.G, self.G, self.G]).cuda()  
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0 + 0.25)
            self._roughness = nn.Parameter(torch.zeros((self._dc.shape[0], 1), device='cuda'))
            vectors = fibonacci_sphere(self.K).to('cuda').unsqueeze(0).repeat(self.V, 1, 1)
            self._sites = nn.Parameter(vectors)
            self._colors = nn.Parameter(torch.full((self.V, self.K, self.color_dim), fill_value=0.25, device='cuda' ))
            self._kappa = nn.Parameter(torch.full((self.V, 1, 1), fill_value=float(init_kappa), device='cuda'))
            self._kappa_levels = torch.tensor([1500, 900, 320, 110, 38.4, 13.4, 4.7, 1.64, 0.57, 0.20], device='cuda', dtype=torch.float32)
            self.bbox_min = torch.tensor([-1.5, -1.5, -1.5], device='cuda')
            self.bbox_max = torch.tensor([1.5, 1.5, 1.5], device='cuda')
            self._language_features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], self.K), device="cuda").requires_grad_(True))
            
            self.mlp_decoder = self.mlp_color = nn.Sequential(*(
                [nn.Linear(self.color_dim*self.topk, hidden_dim), nn.ReLU(True)]
                + sum([[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
                    for _ in range(num_hidden_layers-1)], [])
                + [nn.Linear(hidden_dim, 3)]
            )).to('cuda')
            
        elif self.color_representation == 'lp_opt_grid':
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            vectors = fibonacci_sphere(self.K).to('cuda').unsqueeze(0).repeat(self.V, 1, 1) 
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0 + 0.2)
            #self._pred_normals = nn.Parameter(torch.rand((self._xyz.shape[0], 3), device='cuda'))
            self._roughness = nn.Parameter(torch.zeros((self._dc.shape[0], 1), device='cuda'))
            self._sites = nn.Parameter(vectors)
            self._mask = nn.Parameter((torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")))
            colors = torch.full((self.V, self.K, 3), fill_value=0.25, device='cuda')
            self._colors = nn.Parameter(colors)
            self._language_features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
            map = torch.full((1, 6, 256, 256, 3), fill_value=0.25, device='cuda')
            self.map = nn.Parameter(map)
            self._alpha = nn.Parameter(torch.full((self.V, 1, 1), fill_value=float(0.0), device='cuda'))
            self._kappa_levels = torch.tensor([1500, 900, 320, 110, 38.4, 13.4, 4.7, 1.64, 0.57, 0.20], device='cuda', dtype=torch.float32)
            with torch.no_grad():
                self.bbox_min = self._xyz.min(dim=0)[0]
                self.bbox_max = self._xyz.max(dim=0)[0]
                
            self.offsets = torch.tensor(
                [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
                device='cuda', dtype=torch.float32
            )  
            
            coords = torch.stack(torch.meshgrid(
                torch.arange(self.G, device='cuda'),
                torch.arange(self.G, device='cuda'),
                torch.arange(self.G, device='cuda'),
                indexing='ij'), dim=-1).view(-1, 3)  
            
            self.voxel_centers = (coords + 0.5) / self.G  
            
        elif self.color_representation == 'lp_opt' or self.color_representation == 'lp_opt_d':
            init_kappa = 1
            hidden_dim = 128
            num_hidden_layers = 2
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            vectors = fibonacci_sphere(self.K).to('cuda').unsqueeze(0).repeat(self.V, 1, 1)
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0 + 0.2)
            #self._dc = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
            #self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda())

            self._pred_normals = nn.Parameter(torch.rand((self._xyz.shape[0], 3), device='cuda'))
            self._roughness = nn.Parameter(torch.zeros((self._dc.shape[0], 1), device='cuda'))
            self._sites = nn.Parameter(vectors)
            self._mask = nn.Parameter(inverse_sigmoid((torch.ones((fused_point_cloud.shape[0], 1), device="cuda")).requires_grad_(True) * 0.001))
            if self.color_dim > 3:
                self._colors = nn.Parameter(torch.randn((self.V, self.K, self.color_dim), device='cuda'))
            else:
                colors = torch.full((self.V, self.K, 3), fill_value=0.25, device='cuda')
                self._colors = nn.Parameter(colors + torch.randn_like(colors) * 0)
            self._alpha = nn.Parameter(torch.full((self.V, 1, 1), fill_value=float(0.0), device='cuda'))
            self._kappa_levels = torch.tensor([1500, 900, 320, 110, 38.4, 13.4, 4.7, 1.64, 0.57, 0], device='cuda', dtype=torch.float32)
            #[600, 220, 80, 28, 10, 3.6, 1.30, 0.47, 0.17]

            self.bbox_min = self._xyz.min(dim=0)[0]
            self.bbox_max = self._xyz.max(dim=0)[0]
            self._language_features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
            
            # coords = torch.stack(torch.meshgrid(
            #     torch.arange(self.G, device='cuda'),
            #     torch.arange(self.G, device='cuda'),
            #     torch.arange(self.G, device='cuda'),
            #     indexing='ij'), dim=-1).view(-1, 3)  

            # self.voxel_centers = (coords + 0.5) / self.G  
            
            self.features = nn.Parameter(torch.randn( (self.V, self.feat_dim), device='cuda') * 0.01)
            self.mlp_sites = nn.Sequential(*(
                [nn.Linear(self.feat_dim, hidden_dim), nn.ReLU(True)]
                + sum([[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
                    for _ in range(num_hidden_layers-1)], [])
                + [nn.Linear(hidden_dim, self.K*3)]
            )).to('cuda')
            self.mlp_color = nn.Sequential(*(
                [nn.Linear(self.feat_dim, hidden_dim), nn.ReLU(True)]
                + sum([[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
                    for _ in range(num_hidden_layers-1)], [])
                + [nn.Linear(hidden_dim, self.K*3)]
            )).to('cuda')
            nn.init.constant(self.mlp_color[-1].bias, 0.2)
            
            self.mlp_decoder = nn.Sequential(*(
                [nn.Linear(self.color_dim*self.topk, hidden_dim), nn.ReLU(True)]
                + sum([[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
                    for _ in range(num_hidden_layers-1)], [])
                + [nn.Linear(hidden_dim, 3)]
            )).to('cuda')
            
            
            points = self.bbox_min + (self.bbox_max - self.bbox_min) * torch.rand(self.V, 3, device='cuda')
            radius = 0.5 * (self.bbox_max - self.bbox_min).min()
            #points = fibonacci_sphere(self.V, radius=radius).cuda()
            self.positions = nn.Parameter(points)
            
            self.mlp_normals = nn.Sequential(*(
                [nn.Linear(self.num_pos_freq *2*3 +3, hidden_dim), nn.ReLU(True)]
                + sum([[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
                    for _ in range(num_hidden_layers-1)], [])
                + [nn.Linear(hidden_dim, 3)]
            )).to('cuda')
            
            n_env_sites = 8192*2
            envmap_res = 128
            env_sites = fibonacci_sphere(n_env_sites).to('cuda').unsqueeze(0)
            self._env_sites = nn.Parameter(env_sites)
            self._env_colors = nn.Parameter(torch.full((n_env_sites, 3), fill_value=0.25, device='cuda'))
            self.env_centers = texel_centers_dirs(envmap_res, device='cuda')
            
            
        elif self.color_representation == 'lp_cubemaps':
            self.map_res = 48
            self.NUM_PROBES = 128
            self.bbox_min = torch.tensor([-1.5, -1.5, -1.5], device='cuda')
            self.bbox_max = torch.tensor([1.5, 1.5, 1.5], device='cuda')

            # [V, 6, H, W, 3]
            cubemaps_init = torch.ones((self.NUM_PROBES, 6, self.map_res, self.map_res, 3),
                                    device='cuda', dtype=torch.float32) * 0.5
            cubemaps_init += torch.randn_like(cubemaps_init) * 0.1
            self.probe_cubemaps = nn.Parameter(cubemaps_init.clamp(0, 1))

            self.map = nn.Parameter(torch.full((1, 6, 256, 256, 3), fill_value=0.25, device='cuda'))

            self._alpha = nn.Parameter(torch.zeros((self.NUM_PROBES, 1), device='cuda'))
            points = self.bbox_min + (self.bbox_max - self.bbox_min) * torch.rand(self.NUM_PROBES, 3, device='cuda')
            self.positions = nn.Parameter(points)

            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0 + 0.2)
            self._roughness = nn.Parameter(torch.ones((self._dc.shape[0], 1), device='cuda') * 0)
            self._language_features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 4), device="cuda"))

            # --- Prepara solo i parametri del layout (niente copie di pixel) ---
            V = self.NUM_PROBES
            H = W = self.map_res
            total_faces = V * 6
            faces_per_row = int(math.ceil(math.sqrt(total_faces)))
            num_rows = int(math.ceil(total_faces / faces_per_row))
            self._faces_per_row = faces_per_row
            self._num_rows = num_rows
            self._tile_h = H
            self._tile_w = W

            # Precalcola la permutazione (ordine facce -> posizione nella griglia)
            perm = torch.arange(total_faces, device='cuda', dtype=torch.long)  # [0..V*6-1]
            self._face_perm = perm  # qui è già identità; cambia se vuoi un layout diverso

            # Precalcola il mapping UV (in [0,1]) per (probe_id, face_id)
            # Niente dati da cubemap, solo offsets e scale.
            u_offset = []
            v_offset = []
            for face_idx in range(total_faces):
                row = face_idx // faces_per_row
                col = face_idx % faces_per_row
                u_offset.append(col / faces_per_row)
                v_offset.append(row / num_rows)
            u_offset = torch.tensor(u_offset, device='cuda', dtype=torch.float32)
            v_offset = torch.tensor(v_offset, device='cuda', dtype=torch.float32)
            u_scale = (1.0 / faces_per_row)
            v_scale = (1.0 / num_rows)

            probe_ids = torch.arange(V, device='cuda').repeat_interleave(6)
            face_ids  = torch.arange(6, device='cuda').repeat(V)
            idx_in_grid = (probe_ids * 6 + face_ids)  # uguale a face_idx

            # [V,6,4] con (u_off, v_off, u_scale, v_scale)
            mapping = torch.stack([
                u_offset[idx_in_grid],
                v_offset[idx_in_grid],
                torch.full_like(u_offset[idx_in_grid], u_scale),
                torch.full_like(v_offset[idx_in_grid], v_scale)
            ], dim=-1).view(V, 6, 4)

            # registra come buffer (non Param) – è solo geometria
            self.probe_face_mapping = mapping

            self.max_lod_tile = int(math.log2(self.map_res))      # 6 per 64
            self.max_lod_envmap = int(math.log2(256))   
            

        elif self.color_representation == 'light_probes':
            self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda() * 0 + 0.25)
            self._roughness = nn.Parameter(torch.ones((self._dc.shape[0], 1), device='cuda') * 0)
            self.grid_res = torch.tensor([self.res, self.res, self.res]).cuda()
            self.bbox_min = torch.tensor([-1.5, -1.5, -1.5], device='cuda')
            self.bbox_max = torch.tensor([1.5, 1.5, 1.5], device='cuda')
            # coords = torch.stack(torch.meshgrid(
            #     torch.arange(self.grid_res[0]),
            #     torch.arange(self.grid_res[1]),
            #     torch.arange(self.grid_res[2]),
            #     indexing='ij'), dim=-1).to('cuda')
            # self.voxel_centers = ((coords + 0.5) / self.grid_res) * (self.bbox_max - self.bbox_min) + self.bbox_min
            # self.offsets = torch.tensor([
            #     [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            #     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            # ], device='cuda')
            
            Dx, Dy, Dz = self.grid_res  # oppure D, H, W, ma dai nomi “per assi”
            zz, yy, xx = torch.meshgrid(
                torch.arange(self.grid_res[0], device='cuda'),
                torch.arange(self.grid_res[1], device='cuda'),
                torch.arange(self.grid_res[2], device='cuda'),
                indexing='ij'
            )  # zz=z (D), yy=y (H), xx=x (W)

            idx_xyz = torch.stack([xx, yy, zz], dim=-1)  # (D,H,W,3) in ordine x,y,z

            voxel_size = (self.bbox_max - self.bbox_min) / self.grid_res  # (3,)
            self.voxel_centers = (idx_xyz + 0.5) * voxel_size + self.bbox_min  # (D,H,W,3) xyz
            
            fib = (fibonacci_sphere(self.max_degree) * 5).repeat(self.res**3, 1, 1).reshape(self.res, self.res, self.res, self.max_degree*3)
            fib = fib.permute(3, 0, 1, 2).unsqueeze(0).cuda().contiguous()
            self._sites = nn.Parameter(fib)
            #self._colors = nn.Parameter(torch.rand(*self.grid_res.tolist(), self.max_degree, 3, device='cuda') * 0.01)
            self._colors = nn.Parameter( (torch.ones(1, self.max_degree * 3, self.res, self.res, self.res, device='cuda') * 0.25).cuda().contiguous() )
            self._features = nn.Parameter(torch.randn(1, self.feat_dim, self.res, self.res, self.res, device='cuda') * 0.01)
            self._wi = nn.Parameter(torch.rand(*self.grid_res.tolist(), self.max_degree, 1, device='cuda') * 0)
            self._language_features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], self.max_degree), device="cuda").requires_grad_(True))
            #init_mlp_sites_and_color(self.mlp_sites, self.mlp_color, K=self.max_degree, device='cuda')


    def training_setup(self, training_args, pretrained=None, config=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = []
        if pretrained is None:
            l += [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        
        if self.color_representation == 'shs':
            l += [
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"}
            ]
        
        elif self.color_representation == 'sgs':
            l += [
                    {'params' : [self._mu], 'lr': 1e-5, "name": "mu"},
                    {'params' : [self._alpha], 'lr': 0.0025/25, "name": "alpha"},
                    {'params' : [self._lambda], 'lr': 1e-2, "name": "lambda"},
                    {'params' : [self._dc], 'lr': 0.0025, "name": "dc"},
                ]
        elif self.color_representation == 'cubemap':
            l += [
                {'params' : [self.map], 'lr': 0.001, "name": "map_lp"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "features_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "features_rest"},
                {'params' : [self._dc], 'lr': 0.001, "name": "dc"},
                {'params' : [self._roughness], 'lr': 0.002, "name": "roughness"},
                {'params': self.mlp_decoder.parameters(), 'lr': 0.001, 'name': 'lp_decoder'},
                {'params': self.mlp_normals.parameters(), 'lr': 0.0005, 'name': 'mlp_normals'},
            ]
            
        elif self.color_representation == 'voronoi':
            if config is None:
                l += [
                    # {'params' : [self._sites], 'lr': 0.01, "name": "sites"},
                    # {'params' : [self._lambd], 'lr': 0.000125, "name": "lambd"},
                    {'params' : [self._sites], 'lr': training_args.sites_lr, "name": "sites"},
                    {'params' : [self._lambd], 'lr': training_args.color_lr, "name": "lambd"},
                    {'params': [self._beta], 'lr': 0.0, 'name': 'beta'},
                    {'params' : [self._dc], 'lr': 0.0, "name": "dc"},
                ]
                
            else:
                l += [
                    {'params' : [self._sites], 'lr': config.sites_lr, "name": "sites"},
                    {'params' : [self._lambd], 'lr': config.lambd_lr, "name": "lambd"},
                    {'params': [self._beta], 'lr': config.beta_lr, 'name': 'beta'},
                    {'params' : [self._dc], 'lr': 0.0025, "name": "dc"},
                ]
        elif self.color_representation == 'big_voronoi':
            l += [
                    {'params' : [self._sites], 'lr': 0.001, "name": "sites_lp"},
                    {'params' : [self._colors], 'lr': 0.0005, "name": "colors_lp"},
                    {'params' : [self._dc], 'lr': 0.001, "name": "dc"},
                    {'params' : [self._roughness], 'lr': 0.002, "name": "roughness"},
            ]
        elif self.color_representation == 'voronoi_r':
            l += [
                    {'params' : [self._sites], 'lr': 0.01, "name": "sites"},
                    {'params' : [self._lambd], 'lr': 0.000125, "name": "lambd"},
                    {'params' : [self._dc], 'lr': 0.0025, "name": "dc"},
                    {'params' : [self._roughness], 'lr': 0.002, "name": "roughness"},
                ]
            
        elif self.color_representation == 'light_probes':
            l += [
                {'params' : [self._dc], 'lr': 0.001, "name": "dc"},
                {'params' : [self._roughness], 'lr': 0.002, "name": "roughness"},
                {'params' : [self._sites], 'lr': 0.01, "name": "lp_sites"},
                {'params' : [self._colors], 'lr': 0.005, "name": "lp_colors"},
                {'params' : [self._wi], 'lr': 0.01, "name": "lp_wi"},
                {'params': self.mlp_color.parameters(), 'lr': 0.001, 'name': 'lp_mlp_color'},
                {'params': self.mlp_sites.parameters(), 'lr': 0.0005, 'name': 'lp_mlp_sites'},
                {'params': self.mlp_kappa.parameters(), 'lr': 0.0005, 'name': 'lp_mlp_kappa'},
                {'params': self.light_mlp.parameters(), 'lr': 0.001, 'name': 'lp_mlp_light'},
                {'params': self._features, 'lr': 0.001, 'name': 'lp_feat'},
                {'params': self.decoder.parameters(), 'lr': 0.0005, 'name': 'lp_decoder'},
                {'params': [self._language_features], 'lr': 0.002, "name": "language_features"},
            ]
        
        elif self.color_representation == 'lp_voronoi_cmap':
            l += [
                {'params' : [self._roughness], 'lr': 0.002, "name": "roughness"},
                {'params' : [self._sites], 'lr': 1e-3, "name": "sites_lp"},
                {'params' : [self._colors], 'lr': 0.001, "name": "lambd_lp"},
                {'params' : [self._kappa], 'lr': 0.001, "name": "kappa_lp"},
                {'params' : [self._dc], 'lr': 0.001, "name": "dc"},      
                {'params': [self._language_features], 'lr': 0.002, "name": "language_features"},   
                {'params': self.mlp_decoder.parameters(), 'lr': 0.0005, 'name': 'lp_decoder'},
            ]
            
            
        elif self.color_representation == 'lp_opt_grid':
            l += [
                {'params' : [self.map], 'lr': 0.001, "name": "map_lp"},
                {'params' : [self._roughness], 'lr': 0.002, "name": "roughness"},
                {'params': [self._language_features], 'lr': 0.002, "name": "language_features"},
                {'params' : [self._sites], 'lr': 1e-3, "name": "sites_lp"},
                {'params' : [self._colors], 'lr': 0.005, "name": "lambd_lp"}, # best 0.005
                {'params' : [self._alpha], 'lr': 1e-3, "name": "kappa_lp"},
                {'params' : [self._dc], 'lr': 0.001, "name": "dc"}, #best 0,001
                {'params': [self._features_dc], 'lr': 0.0025, "name": "features_dc"},
                {'params': [self._features_rest], 'lr': 0.0025 / 20.0, "name": "features_rest"},
                # {'params' : [self._mask], 'lr': 0.002, "name": "mask"}
            ]
            
        elif self.color_representation == 'lp_opt' or self.color_representation == 'lp_opt_d':
            l += [
                {'params' : [self._alpha], 'lr': 0.001, "name": "kappa_lp"},
                {'params' : [self._colors], 'lr': 0.00015, "name": "lambd_lp"}, # best 0.005
                {'params' : [self.map], 'lr': 0.002, "name": "map_lp"},
                {'params': [self.positions], 'lr': 0.001, 'name': 'positions_lp'},
                {'params' : [self._roughness], 'lr': 0.006, "name": "roughness"},
                {'params' : [self._sites], 'lr': 0.002, "name": "sites_lp"},
                {'params' : [self._dc], 'lr': 0.0004, "name": "dc"},  #best 0,001
                {'params': [self._language_features], 'lr': 0.002, "name": "language_features"},
                {'params': [self._features_dc], 'lr': 0.0025, "name": "features_dc"},
                {'params': [self._features_rest], 'lr': 0.0025 / 20.0, "name": "features_rest"},
                {'params': [self._env_colors], 'lr': 0.001, "name": "lp_env_colors"},
                {'params': [self._env_sites], 'lr': 0, "name": "lp_env_sites"},
            ]
            
                    
        elif self.color_representation == 'lp_cubemaps':
            l += [
                {'params' : [self.probe_cubemaps], 'lr': 0.0015, "name": "cubeatlas_lp"},
                {'params' : [self._alpha], 'lr': 0.0008, "name": "kappa_lp"},
                {'params' : [self.map], 'lr': 0.001, "name": "map_lp"},
                {'params': [self.positions], 'lr': 0.004, 'name': 'positions_lp'},
                {'params' : [self._roughness], 'lr': 0.007, "name": "roughness"},
                {'params' : [self._dc], 'lr': 0.002, "name": "dc"},  #best 0,001
                {'params': [self._features_dc], 'lr': 0.0025, "name": "features_dc"},
                {'params': [self._features_rest], 'lr': 0.0025 / 20.0, "name": "features_rest"},
                {'params': [self._language_features], 'lr': 0.002, "name": "language_features"}
            ]
        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        # self.positions_scheduler_args = get_expon_lr_func(lr_init=1e-2,
        #                                             lr_final=1e-5,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return
            # elif param_group["name"] == "positions_lp":
            #     lr = self.positions_scheduler_args(iteration)
            #     param_group['lr'] = lr
            

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        self._opacity.data[torch.isnan(self._opacity.data.mean(dim=-1))] = 0.0
        self._opacity.data[torch.isnan(self._xyz.data.mean(dim=-1))] = 0.0
        self._opacity.data[torch.isnan(self._scaling.data.mean(dim=-1))] = 0.0
        self._opacity.data[torch.isnan(self._rotation.data.mean(dim=-1))] = 0.0
        
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    
    def reset_sites(self):
        sites_new = fibonacci_sphere(self.K).to('cuda').unsqueeze(0).repeat(self.V, 1, 1)
        optimizable_tensors = self.replace_tensor_to_optimizer(sites_new, "sites_lp")
        self._sites = optimizable_tensors["sites_lp"]
        
        
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'lp' in group['name']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        
        if self.color_representation == 'shs':
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        elif self.color_representation == 'sgs':
            self._mu = optimizable_tensors['mu']
            self._lambda = optimizable_tensors['lambda']
            self._alpha = optimizable_tensors['alpha']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation == 'voronoi':
            self._sites = optimizable_tensors['sites']
            self._lambd = optimizable_tensors['lambd']
            self._beta = optimizable_tensors['beta']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation == 'voronoi_r':
            self._sites = optimizable_tensors['sites']
            self._lambd = optimizable_tensors['lambd']
            self._roughness = optimizable_tensors['roughness']
        elif self.color_representation == 'big_voronoi' or self.color_representation == 'cubemap':
            self._roughness = optimizable_tensors['roughness']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation in ['light_probes', 'lp_opt', 'lp_cubemaps', 'lp_opt_grid', 'lp_opt_d']:
            self._dc = optimizable_tensors['dc']
            self._roughness = optimizable_tensors['roughness']
            self._language_features = optimizable_tensors['language_features']
            # self._pred_normals = optimizable_tensors['pred_normals']
            # self._mask = optimizable_tensors['mask']

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #assert len(group["params"]) == 1
            if 'lp' in group['name']:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, params):
        # d = {"xyz": new_xyz,
        # "f_dc": new_features_dc,
        # "f_rest": new_features_rest,
        # "opacity": new_opacities,
        # "scaling" : new_scaling,
        # "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        
        if self.color_representation == 'shs':
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        elif self.color_representation == 'sgs':
            self._mu = optimizable_tensors['mu']
            self._lambda = optimizable_tensors['lambda']
            self._alpha = optimizable_tensors['alpha']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation == 'voronoi':
            self._sites = optimizable_tensors['sites']
            self._lambd = optimizable_tensors['lambd']
            self._beta = optimizable_tensors['beta']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation == 'big_voronoi' or self.color_representation == 'cubemap':
            self._roughness = optimizable_tensors['roughness']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation == 'voronoi_r':
            self._sites = optimizable_tensors['sites']
            self._lambd = optimizable_tensors['lambd']
            self._roughness = optimizable_tensors['roughness']
            self._dc = optimizable_tensors['dc']
        elif self.color_representation in ['light_probes', 'lp_opt', 'lp_opt_d', 'lp_cubemaps', 'lp_opt_grid']:
            self._dc = optimizable_tensors['dc']
            self._roughness = optimizable_tensors['roughness']
            self._language_features = optimizable_tensors['language_features']
            # self._pred_normals = optimizable_tensors['pred_normals']
            # self._mask = optimizable_tensors['mask']
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        params = {
            'xyz': new_xyz,
            'opacity': new_opacity,
            'scaling': new_scaling,
            'rotation': new_rotation,
            'features_dc': new_features_dc,
            'features_rest': new_features_rest
        }
        
        if self.color_representation == 'shs':
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            params.update ( {
                'f_dc': new_features_dc,
                'f_rest': new_features_rest
            })
        elif self.color_representation == 'sgs':
            new_mu = self._mu[selected_pts_mask].repeat(N, 1, 1)
            new_lambda = self._lambda[selected_pts_mask].repeat(N, 1, 1)
            new_alpha = self._alpha[selected_pts_mask].repeat(N, 1, 1)
            new_dc = self._dc[selected_pts_mask].repeat(N, 1)
            params.update ( {
                'mu': new_mu,
                'lambda': new_lambda,
                'alpha': new_alpha,
                'dc': new_dc
            })
            
        elif self.color_representation == 'voronoi':
            new_sites = self._sites[selected_pts_mask].repeat(N, 1, 1)
            new_lambd = self._lambd[selected_pts_mask].repeat(N, 1, 1)
            new_beta = self._beta[selected_pts_mask].repeat(N, 1, 1)
            new_dc = self._dc[selected_pts_mask].repeat(N, 1)
            params.update ( {
                'sites': new_sites,
                'lambd': new_lambd,
                'beta': new_beta,
                'dc': new_dc
            })
        elif self.color_representation == 'voronoi_r':
            new_sites = self._sites[selected_pts_mask].repeat(N, 1, 1)
            new_lambd = self._lambd[selected_pts_mask].repeat(N, 1, 1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_dc = self._dc[selected_pts_mask].repeat(N, 1)
            params.update ( {
                'sites': new_sites,
                'lambd': new_lambd,
                'roughness': new_roughness,
                'dc': new_dc
            })
        elif self.color_representation == 'big_voronoi' or self.color_representation == 'cubemap':
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_dc = self._dc[selected_pts_mask].repeat(N, 1)
            params.update ( {
                'roughness': new_roughness,
                'dc': new_dc
            })
        elif self.color_representation in ['light_probes', 'lp_opt', 'lp_opt_d', 'lp_cubemaps', 'lp_opt_grid']:
            new_dc = self._dc[selected_pts_mask].repeat(N, 1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_language_features = self._language_features[selected_pts_mask].repeat(N, 1)
            # new_pred_normals = self._pred_normals[selected_pts_mask].repeat(N, 1)
            # new_mask = self._mask[selected_pts_mask].repeat(N, 1)
            params.update ( {
                'dc': new_dc,
                'roughness': new_roughness,
                'language_features': new_language_features,
                # 'pred_normals': new_pred_normals,
                # 'mask': new_mask,
            })

        self.densification_postfix(params)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        params = {
            'xyz': self._xyz[selected_pts_mask],
            'opacity': self._opacity[selected_pts_mask],
            'scaling': self._scaling[selected_pts_mask],
            'rotation': self._rotation[selected_pts_mask],
            'features_dc': self._features_dc[selected_pts_mask],
            'features_rest': self._features_rest[selected_pts_mask]
            }
        
        if self.color_representation == 'shs':
            params.update( {
                'f_dc': self._features_dc[selected_pts_mask],
                'f_rest': self._features_rest[selected_pts_mask]
            })

        
        elif self.color_representation == 'sgs':
            params.update( {
                'mu': self._mu[selected_pts_mask],
                'alpha': self._alpha[selected_pts_mask],
                'lambda': self._lambda[selected_pts_mask],
                'dc': self._dc[selected_pts_mask]
            })
        
        elif self.color_representation == 'voronoi':
            params.update( {
                'lambd': self._lambd[selected_pts_mask],
                'beta': self._beta[selected_pts_mask],
                'sites': self._sites[selected_pts_mask],
                'dc': self._dc[selected_pts_mask]
            } )
        
        elif self.color_representation == 'voronoi_r':
            params.update( {
                'lambd': self._lambd[selected_pts_mask],
                'roughness': self._roughness[selected_pts_mask],
                'sites': self._sites[selected_pts_mask],
                'dc': self._dc[selected_pts_mask]
            } )
        elif self.color_representation == 'big_voronoi' or self.color_representation == 'cubemap':
            params.update( {
                'roughness': self._roughness[selected_pts_mask],
                'dc': self._dc[selected_pts_mask]
            } )
        
        elif self.color_representation in ['light_probes', 'lp_opt', 'lp_opt_d', 'lp_cubemaps', 'lp_opt_grid']:
            params.update( {
                'dc': self._dc[selected_pts_mask],
                'roughness': self._roughness[selected_pts_mask],
                'language_features': self._language_features[selected_pts_mask],
                # 'pred_normals': self._pred_normals[selected_pts_mask],
                # 'mask': self._mask[selected_pts_mask]
            } )  

        self.densification_postfix(params)
        
        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        
    
    @torch.no_grad()
    def compute_top_idx_env(self,
                        v_chunk: int = 512,
                        c_chunk: int = 2048,
                        s_chunk: int = 2048,
                        kappa=None,
                        use_vmf: bool = False,
                        allow_tf32: bool = True):
        """
        Calcola, per ogni (V,C), gli indici top-K sui siti (S) massimizzando
        <site, center> (eventualmente pesato da κ o con scoring vMF).
        Implementa batching e operazioni in-place per ridurre il picco di memoria.

        Parametri
        ---------
        v_chunk, c_chunk, s_chunk : dimensioni dei chunk per V, C, S
        kappa : None, (S,) o (V,S)
        use_vmf : se True usa score vMF: κ·dot + log κ − log sinh κ
        allow_tf32 : abilita TF32 su GPU Ampere+ per velocizzare la matmul
        """
        # if allow_tf32 and torch.backends.cuda.matmul.is_available():
        #     torch.backends.cuda.matmul.allow_tf32 = True

        print('Print: updating topk idx ...')

        # --- genera i siti normalizzati (V,S,3) ---

        sites = F.normalize(self._env_sites, dim=-1)  # (V,S,3)

        V, S, _ = sites.shape
        C = self.env_centers.shape[0]
        M = self.topk

        dev = sites.device

        # dtype calcolo punteggi: tieni fp32 anche se input è fp16/bf16
        sdtype = torch.float32

        # indici compatti se possibile (dipende da S)
        idx_dtype = torch.int16 if S <= 65535 else torch.int32
        self.env_idx_topk = torch.empty((1, C, M), device=dev, dtype=idx_dtype)

        # κ per-sito (V,S) opzionale
        if kappa is not None:
            if kappa.dim() == 1:   # (S,)
                kappa_vs = kappa[None, :].expand(V, S)
            else:                  # (V,S)
                kappa_vs = kappa
            kappa_vs = kappa_vs.to(sdtype).to(dev)

        centers = self.env_centers.to(sdtype).to(dev)  # nessuna normalizzazione extra

        # --- loop su V ---
        for v0 in range(0, V, v_chunk):
            v1 = min(V, v0 + v_chunk)
            Vb = v1 - v0

            sites_v = sites[v0:v1]                                # (Vb,S,3)
            if sites_v.dtype != sdtype:
                sites_v = sites_v.to(sdtype)

            kappa_v = kappa_vs[v0:v1] if kappa is not None else None

            # --- loop su C ---
            for c0 in range(0, C, c_chunk):
                c1 = min(C, c0 + c_chunk)
                Cb = c1 - c0
                centers_c = centers[c0:c1]                        # (Cb,3)
                centers_c_t = centers_c.T.contiguous()            # (3,Cb)

                # buffer Top-K per questo blocco C: stesso dtype dei punteggi
                best_vals = torch.full((Vb, Cb, M), -float('inf'), device=dev, dtype=sdtype)
                best_idx  = torch.full((Vb, Cb, M), -1, device=dev, dtype=idx_dtype)

                # --- loop su S ---
                for s0 in range(0, S, s_chunk):
                    s1 = min(S, s0 + s_chunk)
                    Sv = s1 - s0

                    # prepara blocchi input già in sdtype (evita cast post-matmul)
                    sites_v_blk = sites_v[:, s0:s1, :].contiguous()          # (Vb,Sv,3)

                    # output pre-allocato per evitare nuova allocazione in matmul
                    scores = torch.empty((Vb, Sv, Cb), device=dev, dtype=sdtype)
                    torch.matmul(sites_v_blk, centers_c_t, out=scores)       # (Vb,Sv,Cb)

                    # applica κ in-place senza materializzare broadcast (Vb,Sv,1)
                    if kappa_v is not None:
                        k_blk = kappa_v[:, s0:s1]                             # (Vb,Sv)
                        k_blk = kappa_v[:, s0:s1]                             # (Vb,Sv)
                        if use_vmf:
                            k_cl = k_blk.clamp_min(1e-12)
                            log_k = k_cl.log()
                            # log(sinh κ) ben condizionato
                            log_sinh_k = torch.where(
                                k_cl > 20.0,
                                k_cl - math.log(2.0),
                                k_cl.sinh().clamp_min(1e-12).log()
                            )
                            scores.mul_(k_blk.unsqueeze(-1))                  # *= κ
                            scores.add_((log_k - log_sinh_k).unsqueeze(-1))   # += bias
                        else:
                            scores.mul_(k_blk)                  # *= κ

                    # Top-K sui siti del blocco S -> (Vb,M,Cb)
                    vals_blk, idx_blk_local = torch.topk(scores, k=M, dim=1, largest=True, sorted=False)

                    # trasponi a (Vb,Cb,M) e aggiungi offset s0 agli indici locali
                    vals_blk = vals_blk.transpose(1, 2).contiguous()
                    idx_blk  = (idx_blk_local.transpose(1, 2).contiguous().to(idx_dtype) + s0)

                    # merge con i migliori correnti: concat su ultima dim e nuovo topk
                    cat_vals = torch.cat([best_vals, vals_blk], dim=-1)       # (Vb,Cb,2M)
                    cat_idx  = torch.cat([best_idx,  idx_blk],  dim=-1)
                    vals_merged, sel = torch.topk(cat_vals, k=M, dim=-1, largest=True, sorted=False)
                    best_idx  = torch.gather(cat_idx,  dim=-1, index=sel)
                    best_vals = vals_merged

                    # libera i blocchi pesanti prima di iterazioni successive
                    del scores, vals_blk, idx_blk, cat_vals, cat_idx, vals_merged, sel, sites_v_blk

                # scrivi risultato per questo blocco C
                self.env_idx_topk[v0:v1, c0:c1, :] = best_idx
                del best_idx, best_vals, centers_c, centers_c_t


        print('Update done')
        return self.env_idx_topk
    
    
    @torch.no_grad()
    def compute_top_idx(self,
                        v_chunk: int = 512,
                        c_chunk: int = 2048,
                        s_chunk: int = 2048,
                        kappa=None,
                        use_vmf: bool = False,
                        allow_tf32: bool = True):
        """
        Calcola, per ogni (V,C), gli indici top-K sui siti (S) massimizzando
        <site, center> (eventualmente pesato da κ o con scoring vMF).
        Implementa batching e operazioni in-place per ridurre il picco di memoria.

        Parametri
        ---------
        v_chunk, c_chunk, s_chunk : dimensioni dei chunk per V, C, S
        kappa : None, (S,) o (V,S)
        use_vmf : se True usa score vMF: κ·dot + log κ − log sinh κ
        allow_tf32 : abilita TF32 su GPU Ampere+ per velocizzare la matmul
        """
        # if allow_tf32 and torch.backends.cuda.matmul.is_available():
        #     torch.backends.cuda.matmul.allow_tf32 = True

        print('Print: updating topk idx ...')

        # --- genera i siti normalizzati (V,S,3) ---
        if getattr(self, 'use_mlp', False):
            enc_p = positional_encoding(self.voxel_centers.reshape(-1, 3), self.num_pos_freq)
            inp = torch.cat([self.features, enc_p], dim=-1)
            sites = F.normalize(self.mlp_sites(self.features).reshape(-1, self.K, 3), dim=-1)
        else:
            sites = F.normalize(self._sites, dim=-1)  # (V,S,3)

        V, S, _ = sites.shape
        C = self.centers.shape[0]
        M = self.topk

        dev = sites.device

        # dtype calcolo punteggi: tieni fp32 anche se input è fp16/bf16
        sdtype = torch.float32

        # indici compatti se possibile (dipende da S)
        idx_dtype = torch.int16 if S <= 65535 else torch.int32
        self.idx_topk = torch.empty((V, C, M), device=dev, dtype=idx_dtype)

        # κ per-sito (V,S) opzionale
        if kappa is not None:
            if kappa.dim() == 1:   # (S,)
                kappa_vs = kappa[None, :].expand(V, S)
            else:                  # (V,S)
                kappa_vs = kappa
            kappa_vs = kappa_vs.to(sdtype).to(dev)

        centers = self.centers.to(sdtype).to(dev)  # nessuna normalizzazione extra

        # --- loop su V ---
        for v0 in range(0, V, v_chunk):
            v1 = min(V, v0 + v_chunk)
            Vb = v1 - v0

            sites_v = sites[v0:v1]                                # (Vb,S,3)
            if sites_v.dtype != sdtype:
                sites_v = sites_v.to(sdtype)

            kappa_v = kappa_vs[v0:v1] if kappa is not None else None

            # --- loop su C ---
            for c0 in range(0, C, c_chunk):
                c1 = min(C, c0 + c_chunk)
                Cb = c1 - c0
                centers_c = centers[c0:c1]                        # (Cb,3)
                centers_c_t = centers_c.T.contiguous()            # (3,Cb)

                # buffer Top-K per questo blocco C: stesso dtype dei punteggi
                best_vals = torch.full((Vb, Cb, M), -float('inf'), device=dev, dtype=sdtype)
                best_idx  = torch.full((Vb, Cb, M), -1, device=dev, dtype=idx_dtype)

                # --- loop su S ---
                for s0 in range(0, S, s_chunk):
                    s1 = min(S, s0 + s_chunk)
                    Sv = s1 - s0

                    # prepara blocchi input già in sdtype (evita cast post-matmul)
                    sites_v_blk = sites_v[:, s0:s1, :].contiguous()          # (Vb,Sv,3)

                    # output pre-allocato per evitare nuova allocazione in matmul
                    scores = torch.empty((Vb, Sv, Cb), device=dev, dtype=sdtype)
                    torch.matmul(sites_v_blk, centers_c_t, out=scores)       # (Vb,Sv,Cb)

                    # applica κ in-place senza materializzare broadcast (Vb,Sv,1)
                    if kappa_v is not None:
                        k_blk = kappa_v[:, s0:s1]                             # (Vb,Sv)
                        k_blk = kappa_v[:, s0:s1]                             # (Vb,Sv)
                        if use_vmf:
                            k_cl = k_blk.clamp_min(1e-12)
                            log_k = k_cl.log()
                            # log(sinh κ) ben condizionato
                            log_sinh_k = torch.where(
                                k_cl > 20.0,
                                k_cl - math.log(2.0),
                                k_cl.sinh().clamp_min(1e-12).log()
                            )
                            scores.mul_(k_blk.unsqueeze(-1))                  # *= κ
                            scores.add_((log_k - log_sinh_k).unsqueeze(-1))   # += bias
                        else:
                            scores.mul_(k_blk)                  # *= κ

                    # Top-K sui siti del blocco S -> (Vb,M,Cb)
                    vals_blk, idx_blk_local = torch.topk(scores, k=M, dim=1, largest=True, sorted=False)

                    # trasponi a (Vb,Cb,M) e aggiungi offset s0 agli indici locali
                    vals_blk = vals_blk.transpose(1, 2).contiguous()
                    idx_blk  = (idx_blk_local.transpose(1, 2).contiguous().to(idx_dtype) + s0)

                    # merge con i migliori correnti: concat su ultima dim e nuovo topk
                    cat_vals = torch.cat([best_vals, vals_blk], dim=-1)       # (Vb,Cb,2M)
                    cat_idx  = torch.cat([best_idx,  idx_blk],  dim=-1)
                    vals_merged, sel = torch.topk(cat_vals, k=M, dim=-1, largest=True, sorted=False)
                    best_idx  = torch.gather(cat_idx,  dim=-1, index=sel)
                    best_vals = vals_merged

                    # libera i blocchi pesanti prima di iterazioni successive
                    del scores, vals_blk, idx_blk, cat_vals, cat_idx, vals_merged, sel, sites_v_blk

                # scrivi risultato per questo blocco C
                self.idx_topk[v0:v1, c0:c1, :] = best_idx
                del best_idx, best_vals, centers_c, centers_c_t


        print('Update done')
        return self.idx_topk
    
    # @torch.no_grad()
    # def compute_top_idx(self,
    #                     c_chunk_init: int = 25_000,   # batch query iniziale
    #                     use_fp16_index: bool = True,
    #                     temp_mem_mb: int = 256):
    #     # sites: (V,S,3) diversi per ogni V
    #     sites = torch.nn.functional.normalize(self._sites, dim=-1)   # (V,S,3)
    #     V, S, _ = sites.shape
    #     C = self.centers.shape[0]
    #     M = self.topk

    #     out_dtype = torch.int16 if S <= 65535 else torch.int32
    #     idx_topk = torch.empty((V, C, M), dtype=out_dtype, device=sites.device)

    #     # Query tutte in CPU float32 contigue
    #     Q_all = self.centers.detach().to(torch.float32).cpu().contiguous().numpy()  # (C,3)

    #     # Risorse FAISS (riusate) con cap sulla temp memory
    #     res = faiss.StandardGpuResources()
    #     if temp_mem_mb is not None:
    #         res.setTempMemory(int(temp_mem_mb) * 1024 * 1024)
    #         # oppure: res.noTempMemory()  # massima stabilità, un filo più lento

    #     cfg = faiss.GpuIndexFlatConfig()
    #     cfg.device = torch.cuda.current_device()
    #     cfg.useFloat16 = bool(use_fp16_index)
    #     cfg.indicesOptions = faiss.INDICES_32_BIT

    #     # Un indice riusato (reset per ogni v)
    #     index = faiss.GpuIndexFlatIP(res, 3, cfg)

    #     for v in range(V):
    #         # libera frammentazione lato Torch prima di popolare FAISS
    #         torch.cuda.synchronize()
    #         torch.cuda.empty_cache()

    #         # DB per questa v (CPU float32 contiguo)
    #         db_v = sites[v].detach().to(torch.float32).cpu().contiguous().numpy()  # (S,3)

    #         index.reset()
    #         index.add(db_v)  # indicizza 1e6×3 in FP16 ⇒ ~6 MB

    #         c = 0
    #         c_chunk = min(c_chunk_init, C)
    #         while c < C:
    #             c1 = min(C, c + c_chunk)
    #             try:
    #                 D, I = index.search(Q_all[c:c1], M)     # (Cb,M)
    #                 idx_topk[v, c:c1] = torch.as_tensor(I, device=idx_topk.device).to(out_dtype)
    #                 c = c1
    #                 # opzionale: se fila liscio più volte, puoi aumentare un po' c_chunk
    #             except RuntimeError as e:
    #                 if "allocMemory" in str(e) or "CUDA memory" in str(e):
    #                     # dimezza batch query e riprova
    #                     new_chunk = max(2048, c_chunk // 2)
    #                     if new_chunk == c_chunk:
    #                         raise
    #                     c_chunk = new_chunk
    #                     torch.cuda.synchronize()
    #                     torch.cuda.empty_cache()
    #                 else:
    #                     raise
    #     torch.cuda.empty_cache()
    #     self.idx_topk = idx_topk
    #     return idx_topk
    # def world_to_voxel_trilin(self, vs, bbox_min, bbox_max, size, align_corners=True):
    #     D,H,W = size
    #     rel = (vs - bbox_min) / (bbox_max - bbox_min)  # [0,1]
    #     if align_corners:
    #         x = rel[:,0]*(W-1); y = rel[:,1]*(H-1); z = rel[:,2]*(D-1)
    #     else:
    #         x = rel[:,0]*W - 0.5; y = rel[:,1]*H - 0.5; z = rel[:,2]*D - 0.5
    #     x0 = torch.floor(x).long(); y0 = torch.floor(y).long(); z0 = torch.floor(z).long()
    #     x1 = (x0+1).clamp(0, W-1); y1 = (y0+1).clamp(0, H-1); z1 = (z0+1).clamp(0, D-1)
    #     x0 = x0.clamp(0, W-1); y0 = y0.clamp(0, H-1); z0 = z0.clamp(0, D-1)

    #     tx = (x - x0.float()).clamp(0,1)
    #     ty = (y - y0.float()).clamp(0,1)
    #     tz = (z - z0.float()).clamp(0,1)

    #     idx8 = torch.stack([
    #         torch.stack([z0,y0,x0], -1),
    #         torch.stack([z0,y0,x1], -1),
    #         torch.stack([z0,y1,x0], -1),
    #         torch.stack([z0,y1,x1], -1),
    #         torch.stack([z1,y0,x0], -1),
    #         torch.stack([z1,y0,x1], -1),
    #         torch.stack([z1,y1,x0], -1),
    #         torch.stack([z1,y1,x1], -1),
    #     ], dim=1)  # (V,8,3)

    #     w000=(1-tx)*(1-ty)*(1-tz); w100=tx*(1-ty)*(1-tz)
    #     w010=(1-tx)*ty*(1-tz);     w110=tx*ty*(1-tz)
    #     w001=(1-tx)*(1-ty)*tz;     w101=tx*(1-ty)*tz
    #     w011=(1-tx)*ty*tz;         w111=tx*ty*tz
    #     w8 = torch.stack([w000,w100,w010,w110,w001,w101,w011,w111], dim=1)  # (V,8)

    #     anchor = w8.argmax(1)  # (V,)
    #     return idx8, w8, anchor
    
    
    def world_to_voxel_trilin(self, vs, bbox_min, bbox_max, size, align_corners=True):
        D, H, W = size
        rel = (vs - bbox_min) / (bbox_max - bbox_min)  # [0,1]
        if align_corners:
            x = rel[:,0]*(W-1); y = rel[:,1]*(H-1); z = rel[:,2]*(D-1)
        else:
            x = rel[:,0]*W - 0.5; y = rel[:,1]*H - 0.5; z = rel[:,2]*D - 0.5

        x0 = torch.floor(x).long(); y0 = torch.floor(y).long(); z0 = torch.floor(z).long()
        x1 = (x0+1).clamp(0, W-1); y1 = (y0+1).clamp(0, H-1); z1 = (z0+1).clamp(0, D-1)
        x0 = x0.clamp(0, W-1);     y0 = y0.clamp(0, H-1);     z0 = z0.clamp(0, D-1)

        tx = (x - x0.float()).clamp(0,1)
        ty = (y - y0.float()).clamp(0,1)
        tz = (z - z0.float()).clamp(0,1)

        idx8 = torch.stack([
            torch.stack([z0,y0,x0], -1),
            torch.stack([z0,y0,x1], -1),
            torch.stack([z0,y1,x0], -1),
            torch.stack([z0,y1,x1], -1),
            torch.stack([z1,y0,x0], -1),
            torch.stack([z1,y0,x1], -1),
            torch.stack([z1,y1,x0], -1),
            torch.stack([z1,y1,x1], -1),
        ], dim=1)  # (N,8,3)

        w000=(1-tx)*(1-ty)*(1-tz); w100=tx*(1-ty)*(1-tz)
        w010=(1-tx)*ty*(1-tz);     w110=tx*ty*(1-tz)
        w001=(1-tx)*(1-ty)*tz;     w101=tx*(1-ty)*tz
        w011=(1-tx)*ty*tz;         w111=tx*ty*tz
        w8 = torch.stack([w000,w100,w010,w110,w001,w101,w011,w111], dim=1)  # (N,8)
        return idx8, w8


    def sample_grid_cubemap_trilin_bilinear(
        self,
        grid_map6: torch.Tensor,   # [R,R,R,6,M,M,3], 2D ordine [v,u]
        idx8: torch.LongTensor,    # [N,8,3] (z,y,x)
        w8: torch.Tensor,          # [N,8]
        dirs: torch.Tensor,        # [N,3]
        *, flip_v: bool = False, align_corners: bool = False,
        bilinear: bool = True
    ) -> torch.Tensor:

        device = grid_map6.device
        dtype  = grid_map6.dtype
        R, M = grid_map6.shape[0], grid_map6.shape[4]
        N = dirs.shape[0]

        def _face_uv01_to_dir(face: torch.Tensor, u01: torch.Tensor, v01: torch.Tensor, *, flip_v: bool=False):
            u = 2.0 * u01 - 1.0
            v = 2.0 * (1.0 - v01) - 1.0 if flip_v else 2.0 * v01 - 1.0

            one  = torch.ones_like(u)
            none = -one

            x = torch.empty_like(u); y = torch.empty_like(v); z = torch.empty_like(u)

            # Convenzione come nel tuo codice (v 'down' mapping per faccia)
            m = (face == 0); x[m] =  one[m];  y[m] = -v[m];  z[m] = -u[m]   # +X
            m = (face == 1); x[m] = none[m];  y[m] = -v[m];  z[m] =  u[m]   # -X
            m = (face == 2); x[m] =  u[m];    y[m] =  one[m]; z[m] =  v[m]  # +Y
            m = (face == 3); x[m] =  u[m];    y[m] = none[m]; z[m] = -v[m]  # -Y
            m = (face == 4); x[m] =  u[m];    y[m] = -v[m];   z[m] =  one[m]# +Z
            m = (face == 5); x[m] = -u[m];    y[m] = -v[m];   z[m] = none[m]# -Z

            d = torch.stack([x, y, z], dim=-1)
            return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        face, uv01 = dirs_to_cubemap_uv01(dirs.to(device).float())
        u01, v01 = uv01.unbind(-1)
        if flip_v:
            v01 = 1.0 - v01

        if align_corners:
            u_cont = u01 * (M - 1)
            v_cont = v01 * (M - 1)
        else:
            u_cont = u01 * M - 0.5
            v_cont = v01 * M - 0.5

        iz = idx8[..., 0].to(device).long()
        iy = idx8[..., 1].to(device).long()
        ix = idx8[..., 2].to(device).long()
        f  = face.to(device).long()
        f_e = f[:, None].expand(N, 8)

        if not bilinear:
            ui = u_cont.round().long().clamp(0, M - 1)
            vi = v_cont.round().long().clamp(0, M - 1)
            ue = ui[:, None].expand(N, 8)
            ve = vi[:, None].expand(N, 8)
            c = grid_map6[iz, iy, ix, f_e, ve, ue].float()     # [N,8,3]
            w8n = w8 / w8.sum(dim=1, keepdim=True).clamp_min(1e-8)
            out = (c * w8n.to(device).unsqueeze(-1).float()).sum(dim=1)
       
        u0 = torch.floor(u_cont).long(); u1 = u0 + 1
        v0 = torch.floor(v_cont).long(); v1 = v0 + 1

        wu = (u_cont - u0.float()).view(N, 1, 1)   # [N,1,1]
        wv = (v_cont - v0.float()).view(N, 1, 1)

        if align_corners:
            u01_00 = u0.float() / (M - 1); v01_00 = v0.float() / (M - 1)
            u01_10 = u1.float() / (M - 1); v01_10 = v0.float() / (M - 1)
            u01_01 = u0.float() / (M - 1); v01_01 = v1.float() / (M - 1)
            u01_11 = u1.float() / (M - 1); v01_11 = v1.float() / (M - 1)
        else:
            u01_00 = (u0.float() + 0.5) / M; v01_00 = (v0.float() + 0.5) / M
            u01_10 = (u1.float() + 0.5) / M; v01_10 = (v0.float() + 0.5) / M
            u01_01 = (u0.float() + 0.5) / M; v01_01 = (v1.float() + 0.5) / M
            u01_11 = (u1.float() + 0.5) / M; v01_11 = (v1.float() + 0.5) / M

        u01_00e = u01_00[:, None].expand(N, 8); v01_00e = v01_00[:, None].expand(N, 8)
        u01_10e = u01_10[:, None].expand(N, 8); v01_10e = v01_10[:, None].expand(N, 8)
        u01_01e = u01_01[:, None].expand(N, 8); v01_01e = v01_01[:, None].expand(N, 8)
        u01_11e = u01_11[:, None].expand(N, 8); v01_11e = v01_11[:, None].expand(N, 8)

        d00 = _face_uv01_to_dir(f_e, u01_00e, v01_00e, flip_v=flip_v)
        d10 = _face_uv01_to_dir(f_e, u01_10e, v01_10e, flip_v=flip_v)
        d01 = _face_uv01_to_dir(f_e, u01_01e, v01_01e, flip_v=flip_v)
        d11 = _face_uv01_to_dir(f_e, u01_11e, v01_11e, flip_v=flip_v)

        f00, uv00 = dirs_to_cubemap_uv01(d00.reshape(-1,3).float())
        f10, uv10 = dirs_to_cubemap_uv01(d10.reshape(-1,3).float())
        f01, uv01p= dirs_to_cubemap_uv01(d01.reshape(-1,3).float())
        f11, uv11 = dirs_to_cubemap_uv01(d11.reshape(-1,3).float())

        f00 = f00.view(N,8); f10 = f10.view(N,8); f01 = f01.view(N,8); f11 = f11.view(N,8)
        u00,v00 = uv00[...,0].view(N,8), uv00[...,1].view(N,8)
        u10,v10 = uv10[...,0].view(N,8), uv10[...,1].view(N,8)
        u01p,v01p= uv01p[...,0].view(N,8), uv01p[...,1].view(N,8)
        u11,v11 = uv11[...,0].view(N,8), uv11[...,1].view(N,8)

        # 3c) uv' -> indici di texel (nearest sui corner; il bilinear userà wu,wv)
        def _uv01_to_indices(u01c, v01c):
            if align_corners:
                u = (u01c * (M - 1)).round().long().clamp(0, M - 1)
                v = (v01c * (M - 1)).round().long().clamp(0, M - 1)
            else:
                u = (u01c * M - 0.5).round().long().clamp(0, M - 1)
                v = (v01c * M - 0.5).round().long().clamp(0, M - 1)
            return u, v

        u00i,v00i = _uv01_to_indices(u00,  v00)
        u10i,v10i = _uv01_to_indices(u10,  v10)
        u01i,v01i = _uv01_to_indices(u01p, v01p)
        u11i,v11i = _uv01_to_indices(u11,  v11)

        # 3d) fetch dei 4 corner (potenzialmente su facce diverse)
        c00 = grid_map6[iz, iy, ix, f00, v00i, u00i].float()
        c10 = grid_map6[iz, iy, ix, f10, v10i, u10i].float()
        c01 = grid_map6[iz, iy, ix, f01, v01i, u01i].float()
        c11 = grid_map6[iz, iy, ix, f11, v11i, u11i].float()

        # 3e) bilinear "vero" con i pesi continui (wu,wv)
        c0  = c00 * (1.0 - wu) + c10 * wu
        c1  = c01 * (1.0 - wu) + c11 * wu
        bil = c0  * (1.0 - wv) + c1  * wv                      # [N,8,3]

        # --- 4) trilineare nel volume
        w8n = w8 / w8.sum(dim=1, keepdim=True).clamp_min(1e-8)
        out = (bil * w8n.to(device).unsqueeze(-1).float()).sum(dim=1)
        return out.to(dtype)


    def fill_octa_from_topk_kappa_per_voxel(self):
        sites   = F.normalize(self._sites,   dim=-1)   # (V,S,3)
        centers = F.normalize(self.centers,  dim=-1)   # (C,3) con C = M*M
        feats   = self._colors                          # (V,S,F)
        M       = self.map_res

        V, S, _ = sites.shape
        Fdim    = feats.shape[-1]
        idx_topk = self.compute_top_idx()               # (V,C,K)
        _, C, K  = idx_topk.shape
        assert C == M*M, f"C={C} != M*M={M*M}"

        sites_exp = sites.unsqueeze(1).expand(V, C, S, 3)
        feats_exp = feats.unsqueeze(1).expand(V, C, S, Fdim)

        dir_ix  = idx_topk.unsqueeze(-1).expand(V, C, K, 3)
        feat_ix = idx_topk.unsqueeze(-1).expand(V, C, K, Fdim)

        s_sel = torch.gather(sites_exp, 2, dir_ix)            # (V,C,K,3)
        f_sel = torch.gather(feats_exp, 2, feat_ix)           # (V,C,K,F)


        
        if self.color_dim > 3:
            grid_oct = torch.sigmoid(self.mlp_decoder(f_sel.reshape(-1, self.color_dim*self.topk)).reshape(V, M, M, 3).contiguous())
        else:
            cos = (s_sel * centers.view(1, C, 1, 3)).sum(-1)      # (V,C,K)
            kappa = self._kappa.view(V,1,1)
            logits = kappa * cos
            w = torch.softmax(logits, dim=-1)                     # (V,C,K)

            out_flat = (w.unsqueeze(-1) * f_sel).sum(dim=-2).reshape(V*M*M, -1)      # (V,C,F)
            grid_oct = out_flat.reshape(V, M, M, 3).contiguous()

        self.grid_oct = grid_oct
        return grid_oct
    

    def fill_cubemaps_from_topk_kappa_per_voxel(self):
        sites   = self._sites            # (V,S,3)
        centers = self.centers           # (C,3)
        feats   = self._colors           # (V,S,F)
        M       = self.map_res

        V, S, _  = sites.shape
        Fdim      = feats.shape[-1]
        idx_topk  = self.compute_top_idx()            # (V,C,K)
        _, C, K   = idx_topk.shape

        sites_exp = sites.unsqueeze(1).expand(V, C, S, 3)
        feats_exp = feats.unsqueeze(1).expand(V, C, S, Fdim)

        dir_ix  = idx_topk.unsqueeze(-1).expand(V, C, K, 3)
        feat_ix = idx_topk.unsqueeze(-1).expand(V, C, K, Fdim)

        s_sel = torch.gather(sites_exp, 2, dir_ix)            # (V,C,K,3)
        f_sel = torch.gather(feats_exp, 2, feat_ix)           # (V,C,K,F)

        cos = (s_sel * centers.view(1, C, 1, 3)).sum(-1)      # (V,C,K)

        kappa = self._kappa.view(-1, 1, 1)                    # (V,1,1)
        logits = kappa * cos
        w = torch.softmax(logits, dim=-1)                     # (V,C,K)

        out_flat = (w.unsqueeze(-1) * f_sel).sum(dim=-2)      # (V,C,F)
        grid_map6 = out_flat.view(V, 6, M, M, Fdim).contiguous()
        self.grid_map6 = grid_map6
        return grid_map6


def build_face_atlas_and_mip(grid_map6: torch.Tensor):
    V, six, M, _, C = grid_map6.shape
    assert six == 6
    T = V * 6
    tiles_x = int(math.ceil(math.sqrt(T)))
    tiles_y = (T + tiles_x - 1) // tiles_x
    pad = tiles_y*tiles_x - T
    faces = grid_map6.reshape(T, M, M, C)
    if pad > 0:
        faces = torch.cat([faces, torch.zeros(pad, M, M, C, device=faces.device, dtype=faces.dtype)], 0)
    atlas = faces.view(tiles_y, tiles_x, M, M, C).permute(0,2,1,3,4).reshape(1, tiles_y*M, tiles_x*M, C).contiguous()
    return atlas, tiles_x, tiles_y, M  # (1,H_at,W_at,C), mip, tiles, M


def build_octa_atlas(grid_oct: torch.Tensor):
    # grid_oct: (V, M, M, C)  (CUDA float32, linear)
    V, M, _, C = grid_oct.shape
    T = V
    tiles_x = int(math.ceil(math.sqrt(T)))
    tiles_y = (T + tiles_x - 1) // tiles_x
    pad = tiles_y * tiles_x - T

    tiles = grid_oct
    if pad > 0:
        tiles = torch.cat([tiles, torch.zeros(pad, M, M, C, device=tiles.device, dtype=tiles.dtype)], 0)

    atlas = tiles.view(tiles_y, tiles_x, M, M, C) \
                 .permute(0, 2, 1, 3, 4) \
                 .reshape(1, tiles_y * M, tiles_x * M, C) \
                 .contiguous()  # (1, H_at, W_at, C)

    return atlas, tiles_x, tiles_y, M

    
def sample_with_face_atlas(
    atlas, tiles_x, tiles_y, M_face,
    idx8_zyx, w8, dirs, roughness, grid_res, *,
    use_r2=True, Wout=1024
):
    D, H, W = grid_res
    N = dirs.shape[0]

    vlin = (idx8_zyx[...,0]*H*W + idx8_zyx[...,1]*W + idx8_zyx[...,2])  # (N,8)

    face, uv01 = dirs_to_cubemap_uv01(F.normalize(dirs, dim=-1).float())
    u01 = uv01[...,0].view(N,1).expand(N,8)
    v01 = uv01[...,1].view(N,1).expand(N,8)
    f8  = face.long().view(N,1).expand(N,8)

    t  = vlin*6 + f8
    tx = (t % tiles_x).float(); ty = (t // tiles_x).float()
    W_at = tiles_x * M_face; H_at = tiles_y * M_face
    u_at = (tx*M_face + u01*M_face) / W_at
    v_at = (ty*M_face + v01*M_face) / H_at

    N8 = u_at.numel()
    Hout = (N8 + Wout - 1) // Wout
    pad = Hout*Wout - N8

    u_flat = u_at.reshape(-1)
    v_flat = v_at.reshape(-1)
    if pad:
        u_flat = torch.cat([u_flat, u_flat.new_zeros(pad)])
        v_flat = torch.cat([v_flat, v_flat.new_zeros(pad)])

    uv = torch.stack([u_flat, v_flat], -1).view(1, Hout, Wout, 2).to(atlas)

    Lmax = 6
    r = roughness.clamp(0,1).squeeze(-1)
    if use_r2: r = r*r
    lod_flat = (r * Lmax).repeat_interleave(8)
    if pad:
        lod_flat = torch.cat([lod_flat, lod_flat.new_zeros(pad)])
    lod_b = lod_flat.view(1, Hout, Wout)

    y = dr.texture(atlas, uv, filter_mode='auto', boundary_mode='clamp',
                   mip=None, mip_level_bias=lod_b, max_mip_level=Lmax)  # (1,Hout,Wout,C)

    y_flat = y.view(-1, y.shape[-1])[:N8]           # (N8,C)
    vals   = y_flat.view(N, 8, y.shape[-1])         # (N,8,C)

    w8n = (w8 / w8.sum(1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    return (vals * w8n.to(vals.dtype)).sum(1) 


def octa_uv_to_dir(uv: torch.Tensor) -> torch.Tensor:
    # uv in [0,1]^2 -> dir (x,y,z) unit
    f = uv * 2.0 - 1.0                   # [-1,1]^2
    x = f[..., 0]
    y = f[..., 1]
    z = 1.0 - x.abs() - y.abs()
    n = torch.stack([x, y, z], dim=-1)   # (..,3)
    mask = z < 0
    x2 = (1.0 - y.abs()) * x.sign()
    y2 = (1.0 - x.abs()) * y.sign()
    n = torch.where(mask.unsqueeze(-1),
                    torch.stack([x2, y2, -z], dim=-1),
                    n)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def octa_texel_centers_dirs(M: int, device='cuda', dtype=torch.float32) -> torch.Tensor:
    # ritorna (C,3) con C = M*M
    r = torch.arange(M, device=device)
    jg, ig = torch.meshgrid(r, r, indexing='ij')
    u = (ig.float() + 0.5) / M
    v = (jg.float() + 0.5) / M
    uv = torch.stack([u, v], dim=-1).reshape(-1, 2)
    d  = octa_uv_to_dir(uv)
    return d.to(device=device, dtype=dtype)


def dir_to_octa_uv(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # d: (...,3) direzioni (qualunque) -> uv: (...,2) in [0,1]
    d = F.normalize(d, dim=-1)                         # sicuro anche se già normalizzato
    a = d.abs().sum(dim=-1, keepdim=True)              # |x|+|y|+|z|
    n = d / a.clamp_min(eps)

    x, y, z = n.unbind(-1)
    mask = z < 0
    x_fold = (1 - y.abs()) * x.sign()
    y_fold = (1 - x.abs()) * y.sign()
    x = torch.where(mask, x_fold, x)
    y = torch.where(mask, y_fold, y)

    u = x * 0.5 + 0.5
    v = y * 0.5 + 0.5
    return torch.stack([u, v], dim=-1)


def sample_with_octa_atlas(
    atlas, tiles_x, tiles_y, M_oct,
    idx8_zyx, w8, dirs, roughness, grid_res, *,
    use_r2=True, Wout=1024
):
    D, H, W = grid_res
    N = dirs.shape[0]

    # voxel id lineare per i corner
    vlin = (idx8_zyx[..., 0] * H * W + idx8_zyx[..., 1] * W + idx8_zyx[..., 2])  # (N,8)
    tx = (vlin % tiles_x).float()
    ty = (vlin // tiles_x).float()

    # UV OCTA della direzione
    uv01 = dir_to_octa_uv(F.normalize(dirs, dim=-1).float())  # (N,2)
    u01 = uv01[..., 0].view(N, 1).expand(N, 8)
    v01 = uv01[..., 1].view(N, 1).expand(N, 8)

    # UV globali nell'atlas
    W_at = tiles_x * M_oct
    H_at = tiles_y * M_oct
    u_at = (tx * M_oct + u01 * M_oct) / W_at
    v_at = (ty * M_oct + v01 * M_oct) / H_at

    # pack UV in [1, Hout, Wout, 2]
    N8 = u_at.numel()
    Hout = (N8 + Wout - 1) // Wout
    pad = Hout * Wout - N8
    u_flat = u_at.reshape(-1); v_flat = v_at.reshape(-1)
    if pad:
        u_flat = torch.cat([u_flat, u_flat.new_zeros(pad)])
        v_flat = torch.cat([v_flat, v_flat.new_zeros(pad)])
    uv = torch.stack([u_flat, v_flat], -1).view(1, Hout, Wout, 2).to(atlas)

    # LOD da roughness -> [1,Hout,Wout]
    Lmax = 7
    r = roughness.clamp(0, 1).squeeze(-1)
    if use_r2: r = r * r
    lod_flat = (r * Lmax).repeat_interleave(8)
    if pad:
        lod_flat = torch.cat([lod_flat, lod_flat.new_zeros(pad)])
    lod_b = lod_flat.view(1, Hout, Wout)

    # sample unico
    y = dr.texture(atlas, uv, filter_mode='auto', boundary_mode='clamp',
                   mip=None, mip_level_bias=lod_b, max_mip_level=Lmax)  # (1,Hout,Wout,C)

    # trilineare nel volume
    y_flat = y.view(-1, y.shape[-1])[:N8]        # (N8,C)
    vals   = y_flat.view(N, 8, y.shape[-1])      # (N,8,C)
    w8n    = (w8 / w8.sum(1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    return (vals * w8n.to(vals.dtype)).sum(1)    # (N,C)



# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

# For inquiries contact  george.drettakis@inria.fr


# import torch
# import numpy as np
# from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
# from torch import nn
# import os
# from utils.system_utils import mkdir_p
# from plyfile import PlyData, PlyElement
# from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
# from utils.graphics_utils import BasicPointCloud
# from utils.general_utils import strip_symmetric, build_scaling_rotation

# class GaussianModel:

#     def setup_functions(self):
#         def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
#             RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
#             trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
#             trans[:,:3,:3] = RS
#             trans[:, 3,:3] = center
#             trans[:, 3, 3] = 1
#             return trans
        
#         self.scaling_activation = torch.exp
#         self.scaling_inverse_activation = torch.log

#         self.covariance_activation = build_covariance_from_scaling_rotation
#         self.opacity_activation = torch.sigmoid
#         self.inverse_opacity_activation = inverse_sigmoid
#         self.rotation_activation = torch.nn.functional.normalize


#     def __init__(self, sh_degree : int):
#         self.active_sh_degree = 0
#         self.max_sh_degree = sh_degree  
#         self._xyz = torch.empty(0)
#         self._features_dc = torch.empty(0)
#         self._features_rest = torch.empty(0)
#         self._scaling = torch.empty(0)
#         self._rotation = torch.empty(0)
#         self._opacity = torch.empty(0)
#         self.max_radii2D = torch.empty(0)
#         self.xyz_gradient_accum = torch.empty(0)
#         self.denom = torch.empty(0)
#         self.optimizer = None
#         self.percent_dense = 0
#         self.spatial_lr_scale = 0
#         self.setup_functions()

#     def capture(self):
#         return (
#             self.active_sh_degree,
#             self._xyz,
#             self._features_dc,
#             self._features_rest,
#             self._scaling,
#             self._rotation,
#             self._opacity,
#             self.max_radii2D,
#             self.xyz_gradient_accum,
#             self.denom,
#             self.optimizer.state_dict(),
#             self.spatial_lr_scale,
#         )
    
#     def restore(self, model_args, training_args):
#         (self.active_sh_degree, 
#         self._xyz, 
#         self._features_dc, 
#         self._features_rest,
#         self._scaling, 
#         self._rotation, 
#         self._opacity,
#         self.max_radii2D, 
#         xyz_gradient_accum, 
#         denom,
#         opt_dict, 
#         self.spatial_lr_scale) = model_args
#         self.training_setup(training_args)
#         self.xyz_gradient_accum = xyz_gradient_accum
#         self.denom = denom
#         self.optimizer.load_state_dict(opt_dict)

#     @property
#     def get_scaling(self):
#         return self.scaling_activation(self._scaling) #.clamp(max=1)
    
#     @property
#     def get_rotation(self):
#         return self.rotation_activation(self._rotation)
    
#     @property
#     def get_xyz(self):
#         return self._xyz
    
#     @property
#     def get_features(self):
#         features_dc = self._features_dc
#         features_rest = self._features_rest
#         return torch.cat((features_dc, features_rest), dim=1)
    
#     @property
#     def get_opacity(self):
#         return self.opacity_activation(self._opacity)
    
#     def get_covariance(self, scaling_modifier = 1):
#         return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

#     def oneupSHdegree(self):
#         if self.active_sh_degree < self.max_sh_degree:
#             self.active_sh_degree += 1

#     def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
#         self.spatial_lr_scale = spatial_lr_scale
#         fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
#         fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
#         features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
#         features[:, :3, 0 ] = fused_color
#         features[:, 3:, 1:] = 0.0

#         print("Number of points at initialisation : ", fused_point_cloud.shape[0])

#         dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
#         scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
#         rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

#         opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

#         self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
#         self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
#         self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
#         self._scaling = nn.Parameter(scales.requires_grad_(True))
#         self._rotation = nn.Parameter(rots.requires_grad_(True))
#         self._opacity = nn.Parameter(opacities.requires_grad_(True))
#         self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

#     def training_setup(self, training_args):
#         self.percent_dense = training_args.percent_dense
#         self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#         self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

#         l = [
#             {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
#             {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
#             {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
#             {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
#             {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
#             {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
#         ]

#         self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
#         self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
#                                                     lr_final=training_args.position_lr_final*self.spatial_lr_scale,
#                                                     lr_delay_mult=training_args.position_lr_delay_mult,
#                                                     max_steps=training_args.position_lr_max_steps)

#     def update_learning_rate(self, iteration):
#         ''' Learning rate scheduling per step '''
#         for param_group in self.optimizer.param_groups:
#             if param_group["name"] == "xyz":
#                 lr = self.xyz_scheduler_args(iteration)
#                 param_group['lr'] = lr
#                 return lr

#     def construct_list_of_attributes(self):
#         l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
#         # All channels except the 3 DC
#         for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
#             l.append('f_dc_{}'.format(i))
#         for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
#             l.append('f_rest_{}'.format(i))
#         l.append('opacity')
#         for i in range(self._scaling.shape[1]):
#             l.append('scale_{}'.format(i))
#         for i in range(self._rotation.shape[1]):
#             l.append('rot_{}'.format(i))
#         return l

#     def save_ply(self, path):
#         mkdir_p(os.path.dirname(path))

#         xyz = self._xyz.detach().cpu().numpy()
#         normals = np.zeros_like(xyz)
#         f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
#         f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
#         opacities = self._opacity.detach().cpu().numpy()
#         scale = self._scaling.detach().cpu().numpy()
#         rotation = self._rotation.detach().cpu().numpy()

#         dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

#         elements = np.empty(xyz.shape[0], dtype=dtype_full)
#         attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
#         elements[:] = list(map(tuple, attributes))
#         el = PlyElement.describe(elements, 'vertex')
#         PlyData([el]).write(path)

#     def reset_opacity(self):
#         opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
#         optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
#         self._opacity = optimizable_tensors["opacity"]

#     def load_ply(self, path):
#         plydata = PlyData.read(path)

#         xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                         np.asarray(plydata.elements[0]["y"]),
#                         np.asarray(plydata.elements[0]["z"])),  axis=1)
#         opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#         features_dc = np.zeros((xyz.shape[0], 3, 1))
#         features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
#         features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
#         features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

#         extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
#         extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
#         assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
#         features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#         for idx, attr_name in enumerate(extra_f_names):
#             features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#         # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
#         features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((xyz.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((xyz.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#         self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#         self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

#         self.active_sh_degree = self.max_sh_degree

#     def replace_tensor_to_optimizer(self, tensor, name):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             if group["name"] == name:
#                 stored_state = self.optimizer.state.get(group['params'][0], None)
#                 stored_state["exp_avg"] = torch.zeros_like(tensor)
#                 stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#         return optimizable_tensors

#     def _prune_optimizer(self, mask):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             stored_state = self.optimizer.state.get(group['params'][0], None)
#             if stored_state is not None:
#                 stored_state["exp_avg"] = stored_state["exp_avg"][mask]
#                 stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#             else:
#                 group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
#                 optimizable_tensors[group["name"]] = group["params"][0]
#         return optimizable_tensors

#     def prune_points(self, mask):
#         valid_points_mask = ~mask
#         optimizable_tensors = self._prune_optimizer(valid_points_mask)

#         self._xyz = optimizable_tensors["xyz"]
#         self._features_dc = optimizable_tensors["f_dc"]
#         self._features_rest = optimizable_tensors["f_rest"]
#         self._opacity = optimizable_tensors["opacity"]
#         self._scaling = optimizable_tensors["scaling"]
#         self._rotation = optimizable_tensors["rotation"]

#         self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

#         self.denom = self.denom[valid_points_mask]
#         self.max_radii2D = self.max_radii2D[valid_points_mask]

#     def cat_tensors_to_optimizer(self, tensors_dict):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             assert len(group["params"]) == 1
#             extension_tensor = tensors_dict[group["name"]]
#             stored_state = self.optimizer.state.get(group['params'][0], None)
#             if stored_state is not None:

#                 stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
#                 stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#             else:
#                 group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
#                 optimizable_tensors[group["name"]] = group["params"][0]

#         return optimizable_tensors

#     def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
#         d = {"xyz": new_xyz,
#         "f_dc": new_features_dc,
#         "f_rest": new_features_rest,
#         "opacity": new_opacities,
#         "scaling" : new_scaling,
#         "rotation" : new_rotation}

#         optimizable_tensors = self.cat_tensors_to_optimizer(d)
#         self._xyz = optimizable_tensors["xyz"]
#         self._features_dc = optimizable_tensors["f_dc"]
#         self._features_rest = optimizable_tensors["f_rest"]
#         self._opacity = optimizable_tensors["opacity"]
#         self._scaling = optimizable_tensors["scaling"]
#         self._rotation = optimizable_tensors["rotation"]

#         self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#         self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#         self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

#     def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
#         n_init_points = self.get_xyz.shape[0]
#         # Extract points that satisfy the gradient condition
#         padded_grad = torch.zeros((n_init_points), device="cuda")
#         padded_grad[:grads.shape[0]] = grads.squeeze()
#         selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
#         selected_pts_mask = torch.logical_and(selected_pts_mask,
#                                               torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

#         stds = self.get_scaling[selected_pts_mask].repeat(N,1)
#         stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
#         means = torch.zeros_like(stds)
#         samples = torch.normal(mean=means, std=stds)
#         rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
#         new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
#         new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
#         new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
#         new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
#         new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
#         new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

#         self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

#         prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
#         self.prune_points(prune_filter)

#     def densify_and_clone(self, grads, grad_threshold, scene_extent):
#         # Extract points that satisfy the gradient condition
#         selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
#         selected_pts_mask = torch.logical_and(selected_pts_mask,
#                                               torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
#         new_xyz = self._xyz[selected_pts_mask]
#         new_features_dc = self._features_dc[selected_pts_mask]
#         new_features_rest = self._features_rest[selected_pts_mask]
#         new_opacities = self._opacity[selected_pts_mask]
#         new_scaling = self._scaling[selected_pts_mask]
#         new_rotation = self._rotation[selected_pts_mask]

#         self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

#     def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
#         grads = self.xyz_gradient_accum / self.denom
#         grads[grads.isnan()] = 0.0

#         self.densify_and_clone(grads, max_grad, extent)
#         self.densify_and_split(grads, max_grad, extent)

#         prune_mask = (self.get_opacity < min_opacity).squeeze()
#         if max_screen_size:
#             big_points_vs = self.max_radii2D > max_screen_size
#             big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
#             prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
#         self.prune_points(prune_mask)

#         torch.cuda.empty_cache()

#     def add_densification_stats(self, viewspace_point_tensor, update_filter):
#         self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
#         self.denom[update_filter] += 1