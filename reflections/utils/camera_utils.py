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

from scene.cameras import Camera
import numpy as np
import torch
import torch.nn.functional as F
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def get_rays(W, H, FoVx, FoVy, R, T):
    fx = fov2focal(FoVx, W)
    fy = fov2focal(FoVy, H)

    K = np.array([
        [fx, 0, 0.5*W],
        [0, fy, 0.5*H],
        [0, 0, 1]
    ])
    
    RT = torch.cat([
        torch.Tensor(np.transpose(R)), 
        torch.Tensor(T)[...,None] 
    ], dim=-1)

    w2c = torch.cat([
        RT,
        torch.Tensor([[0,0,0,1]])
    ], dim=0)
    c2w = np.linalg.inv(w2c)
    c2w = torch.Tensor(c2w)
    c2w[:3, 1:3] *= -1
    
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 5, 6, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    W = gt_image.shape[2]
    H = gt_image.shape[1]    
    rays_o, rays_d = get_rays(W, H, cam_info.FovX, cam_info.FovY, cam_info.R, cam_info.T)
    rays_d = F.normalize(rays_d.cuda(), dim=-1)
    
    return Camera(
        colmap_id=cam_info.uid, 
        R=cam_info.R, T=cam_info.T, 
        FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
        image=gt_image, 
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name, uid=id, data_device=args.data_device,
        rays_o=rays_o, rays_d=rays_d
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry