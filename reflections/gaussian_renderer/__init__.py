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

import torchvision
import torch
import math
import os

from diff_surfel_2dgs import GaussianRasterizationSettings as GaussianRasterizationSettings_2dgs
from diff_surfel_2dgs import GaussianRasterizer as GaussianRasterizer_2dgs
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization_real import GaussianRasterizationSettings as GaussianRasterizationSettings_real
from diff_surfel_rasterization_real import GaussianRasterizer as GaussianRasterizer_real

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
import torch.nn.functional as F


def get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

DIR="result/"


def render_real_df(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, iteration=0, ENV_CENTER=None, ENV_RADIUS=None,
                dump_images=False, idx=-1, config_name='test'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings_2dgs(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    raster_settings_black = GaussianRasterizationSettings_real(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=True,
    )

    rasterizer = GaussianRasterizer_2dgs(raster_settings=raster_settings)
    rasterizer_black = GaussianRasterizer_real(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    
    rendered_image, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    

    rets =  {
        "render": rendered_image,
    }

    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness

    pad = torch.zeros(means3D.shape[0], 1, device='cuda')

    gs_in = torch.ones(pc.get_xyz.shape[0], 1, device='cuda')
    gs_in[get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS)] = 0.0
    
    gs_out = torch.zeros_like(gs_in)
    gs_out[get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS)] = 1.0
    input_ts = torch.cat([gs_roughness, pc.get_xyz, gs_in, gs_out], dim=-1)
    
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = gs_albedo,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    
    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / (render_alpha))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()
    
    viewdirs = viewpoint_camera.rays_d
    normals = render_normal.permute(1,2,0)
    wo = F.normalize(reflect(-viewdirs, normals), dim=-1)
    
    out_ts = out_ts.permute(1,2,0)
    
    albedo_map = albedo_map.permute(1,2,0)
    roughness_map = out_ts[..., :1]
    pos_map = out_ts[..., 1:4]
    in_map = out_ts[..., 4:5]
    
    with torch.no_grad():
        select_index = (in_map.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]

    if len(select_index) > 0:
    
        wo = wo.reshape(-1, 3)[select_index]
        normals = normals.reshape(-1, 3)[select_index]
        roughness_map = roughness_map.reshape(-1, 1)[select_index]
        albedo_map = albedo_map.reshape(-1, 3)[select_index]
        pos_map = pos_map.reshape(-1, 3)[select_index]

        if pc.use_specular:
            spec_light = pc.compute_specular(wo, pos_map, roughness_map) 
            diff_light = albedo_map
            pbr_rgb = spec_light + diff_light
        else:
            diff_light = albedo_map
            pbr_rgb = diff_light 


        output_rgb = torch.zeros(image_height, image_width, 3).cuda()
        output_rgb.reshape(-1, 3)[select_index] = pbr_rgb
        output_rgb = output_rgb.permute(2,0,1)

        ref_w = out_ts[..., 4:5].permute(2,0,1).detach()
        out_w = out_ts[..., 5:6].permute(2,0,1).detach()
        full_rgb = ref_w*output_rgb + out_w*rendered_image
        
    else:
        full_rgb = rendered_image
        ref_w = out_ts[..., 5:6].permute(2,0,1).detach()
        out_w = out_ts[..., 6:7].permute(2,0,1).detach()
    
    rets.update({
        'pbr_rgb': full_rgb,
        'ref_w': ref_w,
        'out_w': out_w,
        'ref_index': get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS),
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'select_index': select_index,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'roughness_map': out_ts[..., :1].repeat(1,1,3).permute(2,0,1)
    }) 
    
    if iteration % 49 == 0:
        with torch.no_grad(): 
            
            os.makedirs(DIR, exist_ok=True)   
            torchvision.utils.save_image(rendered_image, DIR+"rendered_image.png")
            torchvision.utils.save_image(render_alpha, DIR+"render_alpha.png")
            torchvision.utils.save_image(((render_normal+1)/2)*render_alpha, DIR+"render_normal.png")
                            
            p_map = torch.zeros(image_height, image_width, 3).cuda()
            p_map.reshape(-1, 3)[select_index] = pos_map
            p_map = p_map.permute(2,0,1)
            
            x_min, x_max = pc._xyz.min(), pc._xyz.max()
            p_map = (p_map - x_min) / (x_max - x_min + 1e-10)
            torchvision.utils.save_image(p_map, DIR+"feature_image.png")

            torchvision.utils.save_image(out_ts[..., :1].repeat(1,1,3).permute(2,0,1), DIR+"roughness.png")
            torchvision.utils.save_image(out_ts[..., 4:5].repeat(1,1,3).permute(2,0,1), DIR+"in.png")
            torchvision.utils.save_image(out_ts[..., 5:6].repeat(1,1,3).permute(2,0,1), DIR+"out.png")

            if len(select_index) > 0:
                if pc.use_specular:
                    output_spec = torch.zeros(image_height, image_width, 3).cuda()
                    output_spec.reshape(-1, 3)[select_index] = spec_light
                    output_spec = output_spec.permute(2,0,1)
                    torchvision.utils.save_image((output_spec), DIR+"spec_light.png")

                output_diff = torch.zeros(image_height, image_width, 3).cuda()
                output_diff.reshape(-1, 3)[select_index] = diff_light
                output_diff = output_diff.permute(2,0,1)
                

                torchvision.utils.save_image(output_rgb*render_alpha, DIR+"pbr_rgb.png")
                torchvision.utils.save_image(full_rgb, DIR+"full_rgb.png")
                torchvision.utils.save_image((output_diff), DIR+"diff_light.png")
                
    if dump_images and pc.use_specular:
            os.makedirs(os.path.join(pc.model_path, config_name, 'albedo'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'specular'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'roughness'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'normals'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'positions'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'outputs'), exist_ok=True)
            
            
            albedo_map = torch.zeros(image_height, image_width, 3).cuda()
            albedo_map.reshape(-1, 3)[select_index] = diff_light
            albedo_map = albedo_map.permute(2,0,1)
            
            specular_map = torch.zeros(image_height, image_width, 3).cuda()
            specular_map.reshape(-1, 3)[select_index] = spec_light
            specular_map = specular_map.permute(2,0,1)
            
            
            rg_map = torch.zeros(image_height, image_width, 3).cuda()
            rg_map.reshape(-1, 3)[select_index] = roughness_map
            rg_map = rg_map.permute(2,0,1)
            
            
            p_map = torch.zeros(image_height, image_width, 3).cuda()
            p_map.reshape(-1, 3)[select_index] = pos_map
            p_map = p_map.permute(2,0,1)
            
            x_min, x_max = pc._xyz.min(), pc._xyz.max()
            p_map = (p_map - x_min) / (x_max - x_min + 1e-10)
            
            torchvision.utils.save_image(((render_normal+1)/2)*render_alpha, os.path.join(pc.model_path, config_name, 'normals', f'{idx}.png'))
            torchvision.utils.save_image((albedo_map), os.path.join(pc.model_path, config_name, 'albedo', f'{idx}.png'))
            torchvision.utils.save_image((specular_map),os.path.join(pc.model_path, config_name, 'specular', f'{idx}.png'))
            torchvision.utils.save_image((rg_map), os.path.join(pc.model_path, config_name, 'roughness', f'{idx}.png'))
            torchvision.utils.save_image((p_map), os.path.join(pc.model_path, config_name, 'positions', f'{idx}.png'))
            torchvision.utils.save_image(full_rgb, os.path.join(pc.model_path, config_name, 'outputs', f'{idx}.png'))

    return rets


def render_df(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, config_name='results', dump_images=False, idx=1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    rets = {}
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=True
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    albedo = pc.get_albedo
    pos = pc.get_xyz
    roughness = pc.get_roughness
    
    input_ts = torch.cat([roughness, pos], dim=-1)
    
    albedo_map, out_ts, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = albedo,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
        
    #get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    out_ts = out_ts.permute(1,2,0)
    
    albedo_map = albedo_map.permute(1,2,0)
    roughness_map = out_ts[..., :1]
    pos_map = out_ts[..., 1:]
    
    
    normals = render_normal.permute(1,2,0)
    viewdirs = viewpoint_camera.rays_d
    wo = F.normalize(reflect(-viewdirs, normals), dim=-1)
    
    with torch.no_grad():
        select_index = (render_alpha.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
    
    wo = wo.reshape(-1, 3)[select_index]

    roughness_map = roughness_map.reshape(-1, 1)[select_index]
    albedo_map = albedo_map.reshape(-1, 3)[select_index]
    viewdirs = viewdirs.reshape(-1, 3)[select_index]

    spec_light = None
    diff_light = albedo_map
    if  pc.use_specular:
        pos_map = pos_map.reshape(-1, 3)[select_index]
        spec_light = pc.compute_specular(wo, pos_map, roughness_map)
        diff_light = albedo_map
        pbr_rgb =  spec_light + diff_light
    else:
        pbr_rgb = albedo_map

    
    output_rgb = torch.zeros(image_height, image_width, 3).cuda()
    output_rgb.reshape(-1, 3)[select_index] = pbr_rgb
    output_rgb = output_rgb.permute(2,0,1)
    

    rg_map = torch.zeros(image_height, image_width, 3).cuda()
    rg_map.reshape(-1, 3)[select_index] = roughness_map
    rg_map = rg_map.permute(2,0,1)
    
    
    albedo_map = torch.zeros(image_height, image_width, 3).cuda()
    albedo_map.reshape(-1, 3)[select_index] = diff_light
    albedo_map = albedo_map.permute(2,0,1)
    

    if spec_light is not None:
        specular_map = torch.zeros(image_height, image_width, 3).cuda()
        specular_map.reshape(-1, 3)[select_index] = spec_light
        specular_map = specular_map.permute(2,0,1)
    else:
        specular_map = None
    
    rets.update({
        'render': output_rgb,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "roughness_map": roughness_map,
        "select_index": select_index,
        'pred_normals': None,
        'opacities': opacity,
        'roughness_map': rg_map,
        'specular_map': (specular_map) if specular_map is not None else None,
        'albedo_map': albedo_map
    }) 
    
    
    with torch.no_grad():
        
        if dump_images:
            
            os.makedirs(os.path.join(pc.model_path, config_name, 'albedo'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'specular'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'roughness'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'normals'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'positions'), exist_ok=True)
            os.makedirs(os.path.join(pc.model_path, config_name, 'outputs'), exist_ok=True)
            
            
            out = output_rgb * render_alpha + (1-render_alpha) * bg_color[:, None, None]
            albedo_map = torch.zeros(image_height, image_width, 3).cuda()
            albedo_map.reshape(-1, 3)[select_index] = diff_light
            albedo_map = albedo_map.permute(2,0,1)
            
            render_normal = F.normalize(render_normal, dim=0)
            torchvision.utils.save_image(((render_normal+1)/2) * render_alpha, os.path.join(pc.model_path, config_name, 'normals', f'{idx}.png'))
            torchvision.utils.save_image(albedo_map, os.path.join(pc.model_path, config_name, 'albedo', f'{idx}.png'))
            torchvision.utils.save_image(out, os.path.join(pc.model_path, config_name, 'outputs', f'{idx}.png'))
            
            
            if pc.use_specular:
                specular_map = torch.zeros(image_height, image_width, 3).cuda()
                specular_map.reshape(-1, 3)[select_index] = spec_light
                specular_map = specular_map.permute(2,0,1)
                
                rg_map = torch.zeros(image_height, image_width, 3).cuda()
                rg_map.reshape(-1, 3)[select_index] = roughness_map
                rg_map = rg_map.permute(2,0,1)
                
                p_map = torch.zeros(image_height, image_width, 3).cuda()
                p_map.reshape(-1, 3)[select_index] = pos_map
                p_map = p_map.permute(2,0,1)
                
                x_min, x_max = pc._xyz.min(), pc._xyz.max()
                p_map = (p_map - x_min) / (x_max - x_min + 1e-10)
                

                torchvision.utils.save_image(specular_map,os.path.join(pc.model_path, config_name, 'specular', f'{idx}.png'))
                torchvision.utils.save_image(rg_map, os.path.join(pc.model_path, config_name, 'roughness', f'{idx}.png'))
                torchvision.utils.save_image(p_map, os.path.join(pc.model_path, config_name, 'positions', f'{idx}.png'))
            
    return rets
