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
import sv_probes 


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
    return pts  


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


    def __init__(self, map_res=32, num_probes=128, num_sites=2048, nn_k=8, topk=8):
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
        self.feat_dim = 16
        self.num_pos_freq = 4
        self.use_mlp = False
        self.nn_k = nn_k
        self.topk = topk

        # Light probes params
        self.color_representation = 'light_probes'
        self.bbox_min = torch.empty(0)
        self.bbox_max = torch.empty(0)
        self.mip_levels = 9
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
        
        self.map = torch.empty(0)
        self.setup_functions()
        
       # nn.Parameter(torch.full((1, 6, 256, 256, self.num_channels), fill_value=0.25, device='cuda'))


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
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,

            self._dc,
            self._roughness,

            self._sites,
            self._colors,
            self._alpha,
            self._kappa_levels,
            self.map,
            self.positions,
            self.idx_topk if hasattr(self, "idx_topk") else None,
        )

    
    def restore(self, model_args, training_args):
        (
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
            opt_state,
            self.spatial_lr_scale,

            self._dc,
            self._roughness,

            self._sites,
            self._colors,
            self._alpha,
            self._kappa_levels,
            self.map,
            self.positions,
            self.idx_topk,
        ) = model_args

        self.training_setup(training_args)

        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
    
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
        return self._dc
    
    def eval(self):
        self.training = False
        self.sites_nn = F.normalize(self._sites, dim=-1)
    
    def train(self):
        self.training = True
        
        
    def eval_envmap(self, wo, roughness=None):
        ''' eval envmap at wo (far-field)'''
        tex = self.map   # [1,6,Hf,Wf,C]
        C = tex.shape[-1]
        device = wo.device
        dtype  = wo.dtype

        N = wo.shape[0]
        Wimg = int(math.ceil(math.sqrt(N)))
        Himg = int(math.ceil(N / Wimg))
        total = Himg * Wimg

        if total != N:
            pad = total - N
            pad_dirs = torch.zeros(pad, 3, device=device, dtype=dtype)
            pad_dirs[:, 2] = 1.0  # +Z
            wo = torch.cat([wo, pad_dirs], dim=0)

        wo_img = wo.view(1, Himg, Wimg, 3).contiguous()

        if roughness is None: #just for testing purposes
            
            out = dr.texture(
                tex, wo_img,
                boundary_mode='cube',
                filter_mode='linear'  
            )  # [1,Himg,Wimg,C]
        else:
            if total != N:
                pad = total - N
                pad_r = torch.ones(pad, 1, device=device, dtype=roughness.dtype)
                roughness = torch.cat([roughness, pad_r], dim=0)
                
            max_lod = torch.log2(torch.as_tensor(tex.shape[2], dtype=torch.float32, device=device))
            lod_img = (roughness.squeeze(-1).clamp(0, 1)) * max_lod
            lod_img = lod_img.view(1, Himg, Wimg)  

            out = dr.texture(
                tex, wo_img,
                boundary_mode='cube',
                filter_mode='linear-mipmap-linear',
                mip_level_bias=lod_img,
                max_mip_level=getattr(self, 'mip_levels', None)  
            )  # [1,Himg,Wimg,C]

        out = out.view(-1, C)[:N, :]

        if getattr(self, 'num_channels', C) > 3:
            out = self.mlp_decoder(out)
        return out

        
    def compute_specular(self, wo, pos, roughness):
        R = self.map_res
        M = self.topk
        B = pos.shape[0]
        V, S, _ = self._sites.shape
        nn_k = self.nn_k
        idx_topk = self.idx_topk
        pos_ = pos.unsqueeze(0)              
        vox_ = self.positions.unsqueeze(0)       

        knn = knn_points(pos_, vox_, K=nn_k, return_nn=False)    
        dk  = knn.dists.squeeze(0).sqrt()           
        iidx = knn.idx.squeeze(0)                              

        # wendland C^2 kernel (found it to be more stable than inverse distance weighting)
        alpha = 1
        h = alpha * dk[:, [-1]]                                  
        q = dk / (h + 1e-8)                                      
        w = (torch.clamp(1.0 - q, min=0.0)**4) * (4.0*q + 1.0)  

        wsum = w.sum(dim=1, keepdim=True)
        weights = w / wsum.clamp_min(1e-8)
        
        self.chosen_probes = iidx
        self.weights = weights
        
        face, u, v = dir_to_face_uv(wo)                             
        i_tex, j_tex = uv_to_ij(u, v, R)
        cell_id = face_ij_to_cellid(face, i_tex, j_tex, R)         
        cell_b = cell_id.unsqueeze(1).expand(-1, nn_k)                 
        cand_idx = idx_topk[iidx, cell_b, :]         
        
        K_levels = self._kappa_levels
        N = int(K_levels.shape[0])
        
        # temp computation 
        r = roughness.reshape(-1).clamp_min(1e-4)                                                            
        lod = ((N - 1) * r).clamp(0, N - 1)                       
        l0 = lod.floor().long()                                       
        l1 = torch.clamp(l0 + 1, max=N - 1)                           
        tL = (lod - l0.float()).unsqueeze(-1).unsqueeze(-1)           
        kL0 = K_levels[l0].view(B, 1, 1)
        kL1 = K_levels[l1].view(B, 1, 1)
        k_level = ((1.0 - tL) * kL0 + tL * kL1).clamp(min=1e-4) 
        
        out_env = self.eval_envmap(wo, roughness) # and this is c_far                  
        
        if self.training == False: # ToDO: backward pass
            out_probes, alphas = sv_probes.forward(self.sites_nn, self._colors, self._alpha.flatten(), idx_topk, iidx,
                                                  weights, wo, k_level.flatten(), cell_id)
            alphas = alphas.unsqueeze(-1)
        else:
            sites_flat  = self._sites.reshape(V*S, 3).contiguous()
            colors_flat = self._colors.reshape(V*S, 3).contiguous()
            alphas_flat = self._alpha.reshape(V, 1).contiguous()
            
            lin = (iidx.unsqueeze(-1) * S + cand_idx).reshape(-1).to(torch.int64)     
            S_q = sites_flat.index_select(0, lin).reshape(B, nn_k, M, 3)  
            C_q = colors_flat.index_select(0, lin).reshape(B, nn_k, M, 3)  
            A_q = alphas_flat.index_select(0, iidx.reshape(-1)).reshape(B, nn_k, 1)
            
            if self.training is not False:
                S_q = F.normalize(S_q, dim=-1)
            A_q = torch.sigmoid(A_q)
            
            scores = (S_q * wo.unsqueeze(1).unsqueeze(1)).sum(dim=-1) * k_level   
            W = F.softmax(scores, dim=-1)                                             
            col_8 = (W.unsqueeze(-1) * C_q).sum(dim=2)                             

            out_probes = (col_8 * weights.unsqueeze(-1)).sum(dim=1)   # this is c_near
            alphas = (A_q * weights.unsqueeze(-1)).sum(dim=1) 
        
        return (alphas * out_env) + ((1 - alphas) * out_probes) # final blending


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
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        vectors = fibonacci_sphere(self.K).to('cuda').unsqueeze(0).repeat(self.V, 1, 1)
        colors = torch.full((self.V, self.K, 3), fill_value=0.25, device='cuda')  
              
        #self._dc = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        self._dc = nn.Parameter(torch.full((fused_point_cloud.shape[0], 3), fill_value=0.2, device="cuda"))
        self._roughness = nn.Parameter(torch.zeros((self._dc.shape[0], 1), device='cuda'))
        self._sites = nn.Parameter(vectors)
        self._colors = nn.Parameter(colors + torch.randn_like(colors) * 0)
        self._alpha = nn.Parameter(torch.full((self.V, 1, 1), fill_value=float(0.0), device='cuda'))
        self._kappa_levels = torch.tensor([1500, 900, 320, 110, 38.4, 13.4, 4.7, 1.64, 0.57, 0], device='cuda', dtype=torch.float32)
        self.map = nn.Parameter(torch.full((1, 6, 256, 256, 3), fill_value=0.25, device='cuda'))
        
        self.bbox_min = self._xyz.min(dim=0)[0]
        self.bbox_max = self._xyz.max(dim=0)[0]
        points = self.bbox_min + (self.bbox_max - self.bbox_min) * torch.rand(self.V, 3, device='cuda')
        self.positions = nn.Parameter(points)


    def training_setup(self, training_args, pretrained=None, config=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
            

        l += [
            {'params' : [self._alpha], 'lr': training_args.alpha_lr, "name": "kappa_lp"},
            {'params' : [self._colors], 'lr': training_args.colors_lr, "name": "lambd_lp"}, 
            {'params' : [self.map], 'lr': training_args.map_lr, "name": "map_lp"},
            {'params': [self.positions], 'lr': training_args.positions_lp_lr, 'name': 'positions_lp'},
            {'params' : [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params' : [self._sites], 'lr': training_args.sites_lr, "name": "sites_lp"},
            {'params' : [self._dc], 'lr': training_args.dc_lr, "name": "dc"},  
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "features_dc"}, # SHs are used only in the real setting for a fair comparison with ref-gs
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "features_rest"},
        ]
            

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return
            

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')

        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')

        l += ['dc_0', 'dc_1', 'dc_2']
        l += ['roughness']

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

        N = xyz.shape[0]


        if hasattr(self, "_dc") and self._dc.numel() > 0:
            dc = self._dc.detach().reshape(N, -1).contiguous().cpu().numpy()
            if dc.shape[1] < 3:
                dc = np.pad(dc, ((0,0),(0,3-dc.shape[1])), mode='constant')
            dc = dc[:, :3]
        else:
            dc = np.zeros((N, 3), dtype=np.float32)

        if hasattr(self, "_roughness") and self._roughness.numel() > 0:
            rough = self._roughness.detach().reshape(N, -1).contiguous().cpu().numpy()
            rough = rough[:, :1]
        else:
            rough = np.zeros((N, 1), dtype=np.float32)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(N, dtype=dtype_full)

        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, dc, rough),
            axis=1
        )
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


        pt_path = os.path.splitext(path)[0] + ".pt"

        sidecar = {
            "_sites": self._sites.detach().cpu() if hasattr(self, "_sites") else None,
            "_colors": self._colors.detach().cpu() if hasattr(self, "_colors") else None,
            "_alpha": self._alpha.detach().cpu() if hasattr(self, "_alpha") else None,
            "_kappa_levels": self._kappa_levels.detach().cpu() if hasattr(self, "_kappa_levels") else None,
            "idx_topk": getattr(self, "idx_topk", None).detach().cpu() if getattr(self, "idx_topk", None) is not None else None,
            "map": self.map.detach().cpu() if hasattr(self, "map") else None,
            "positions": self.positions.detach().cpu() if hasattr(self, "positions") else None,

            "map_res": getattr(self, "map_res", None),
            "mip_levels": getattr(self, "mip_levels", None),
            "V": getattr(self, "V", None),
            "K": getattr(self, "K", None),
            "nn_k": getattr(self, "nn_k", None),
            "topk": getattr(self, "topk", None),
        }

        torch.save(sidecar, pt_path)



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
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sh_rest = (self.max_sh_degree + 1) ** 2 - 1
        if features_extra.shape[1] == 3 * sh_rest:
            features_extra = features_extra.reshape((features_extra.shape[0], 3, sh_rest))
        else:
            features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        device = "cuda"

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        props = {p.name for p in plydata.elements[0].properties}

        if {"dc_0", "dc_1", "dc_2"}.issubset(props):
            dc = np.stack([
                np.asarray(plydata.elements[0]["dc_0"]),
                np.asarray(plydata.elements[0]["dc_1"]),
                np.asarray(plydata.elements[0]["dc_2"]),
            ], axis=1).astype(np.float32)
            self._dc = nn.Parameter(torch.tensor(dc, dtype=torch.float, device=device).requires_grad_(True))
        else:
            self._dc = nn.Parameter(torch.zeros((xyz.shape[0], 3), device=device).requires_grad_(True))

        if "roughness" in props:
            rough = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis].astype(np.float32)
            self._roughness = nn.Parameter(torch.tensor(rough, dtype=torch.float, device=device).requires_grad_(True))
        else:
            self._roughness = nn.Parameter(torch.zeros((xyz.shape[0], 1), device=device).requires_grad_(True))

        pt_path = os.path.splitext(path)[0] + ".pt"
        if os.path.exists(pt_path):
            sidecar = torch.load(pt_path, map_location=device)

            if sidecar.get("_sites", None) is not None:
                self._sites = nn.Parameter(sidecar["_sites"].to(device).requires_grad_(True))
            if sidecar.get("_colors", None) is not None:
                self._colors = nn.Parameter(sidecar["_colors"].to(device).requires_grad_(True))
            if sidecar.get("_alpha", None) is not None:
                self._alpha = nn.Parameter(sidecar["_alpha"].to(device).requires_grad_(True))
            if sidecar.get("_kappa_levels", None) is not None:
                self._kappa_levels = sidecar["_kappa_levels"].to(device)

            if sidecar.get("idx_topk", None) is not None:
                self.idx_topk = sidecar["idx_topk"].to(device)
            else:
                self.idx_topk = None

            if sidecar.get("map", None) is not None:
                self.map = nn.Parameter(sidecar["map"].to(device).requires_grad_(True))
            if sidecar.get("positions", None) is not None:
                self.positions = nn.Parameter(sidecar["positions"].to(device).requires_grad_(True))

            self.map_res = sidecar.get("map_res", getattr(self, "map_res", 32))
            self.mip_levels = sidecar.get("mip_levels", getattr(self, "mip_levels", 9))
            self.V = sidecar.get("V", getattr(self, "V", None))
            self.K = sidecar.get("K", getattr(self, "K", None))
            self.nn_k = sidecar.get("nn_k", getattr(self, "nn_k", None))
            self.topk = sidecar.get("topk", getattr(self, "topk", None))

            self.centers = texel_centers_dirs(self.map_res, device=device, dtype=torch.float32)

        if getattr(self, "training", False) is False and hasattr(self, "_sites") and self._sites.numel() > 0:
            self.sites_nn = F.normalize(self._sites, dim=-1)


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
        self._dc = optimizable_tensors['dc']
        self._roughness = optimizable_tensors['roughness']
        
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

        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        
        self._dc = optimizable_tensors['dc']
        self._roughness = optimizable_tensors['roughness']

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
        
        new_dc = self._dc[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)

        params.update ( {
            'dc': new_dc,
            'roughness': new_roughness,
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
        
        params.update( {
            'dc': self._dc[selected_pts_mask],
            'roughness': self._roughness[selected_pts_mask],

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
    def compute_top_idx(self,
                        v_chunk: int = 512,
                        c_chunk: int = 2048,
                        s_chunk: int = 2048,
                        ):
        '''For each probe and each directional cell, we select the top-k site 
           directions with maximum dot product to the cell center, providing 
           a per-cell candidate set for efficient angular lookup.'''
        
        print('Print: updating topk idx ...')

        sites = F.normalize(self._sites, dim=-1) 

        V, S, _ = sites.shape
        C = self.centers.shape[0]
        M = self.topk
        dev = sites.device
        self.idx_topk = torch.empty((V, C, M), device=dev, dtype=torch.int64)
        centers = self.centers

        for v0 in range(0, V, v_chunk):
            v1 = min(V, v0 + v_chunk)
            Vb = v1 - v0

            sites_v = sites[v0:v1]                                

            for c0 in range(0, C, c_chunk):
                c1 = min(C, c0 + c_chunk)
                Cb = c1 - c0
                centers_c = centers[c0:c1]                        
                centers_c_t = centers_c.T.contiguous()            
              
                best_vals = torch.full((Vb, Cb, M), -float('inf'), device=dev)
                best_idx  = torch.full((Vb, Cb, M), -1, device=dev)
                
                for s0 in range(0, S, s_chunk):
                    s1 = min(S, s0 + s_chunk)
                    Sv = s1 - s0
                    sites_v_blk = sites_v[:, s0:s1, :].contiguous()          

                    scores = torch.empty((Vb, Sv, Cb), device=dev)
                    torch.matmul(sites_v_blk, centers_c_t, out=scores)       

                    vals_blk, idx_blk_local = torch.topk(scores, k=M, dim=1, largest=True, sorted=False)

                    vals_blk = vals_blk.transpose(1, 2).contiguous()
                    idx_blk  = (idx_blk_local.transpose(1, 2).contiguous() + s0)

                    cat_vals = torch.cat([best_vals, vals_blk], dim=-1)       # (Vb,Cb,2M)
                    cat_idx  = torch.cat([best_idx,  idx_blk],  dim=-1)
                    vals_merged, sel = torch.topk(cat_vals, k=M, dim=-1, largest=True, sorted=False)
                    best_idx  = torch.gather(cat_idx,  dim=-1, index=sel)
                    best_vals = vals_merged

                    del scores, vals_blk, idx_blk, cat_vals, cat_idx, vals_merged, sel, sites_v_blk

                self.idx_topk[v0:v1, c0:c1, :] = best_idx
                del best_idx, best_vals, centers_c, centers_c_t


        print('Update done')
        return self.idx_topk
    
    


   