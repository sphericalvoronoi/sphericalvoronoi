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
import numpy as np
from PIL import Image
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, entropy_loss
from gaussian_renderer import network_gui, render_df, render_real_df
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import json
import torch.nn.functional as F
import torchvision
import re
import torch.nn as nn

TENSORBOARD_FOUND = False


def write(string):
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
        print(string, file=f)
        
        
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(num_probes=args.num_probes, num_sites=args.num_sites, map_res=args.map_res)

    scene = Scene(dataset, gaussians, resolution_scales=[1.0])
    
    gaussians.training_setup(opt)
    
    gaussians.use_light_probes = False
    gaussians.use_specular = False
    gaussians.model_path = args.model_path
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    print('Training set length', len(viewpoint_stack))
        
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    

    ENV_CENTER = torch.tensor([float(c) for c in args.env_scope_center], device='cuda')
    ENV_RADIUS = args.env_scope_radius
    XYZ = [int(float(c)) for c in args.xyz_axis]
    
    print(ENV_CENTER, ENV_RADIUS, XYZ)
    
    render_fun = render_real_df
    gaussians.train()
    for iteration in range(first_iter, opt.iterations + 1):     
        
        if iteration == 28_000:
            for param in gaussians.optimizer.param_groups:
                if param['name'] in ['sites_lp', 'positions_lp']: 
                    param['lr'] = 0   

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        data_idx = np.random.randint(len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack[data_idx]
        
        bg = torch.zeros((3), device="cuda") 
        
        
        if iteration == 1 or iteration % 501 == 0:
            gaussians.compute_top_idx()
            
        if iteration == args.warmup_iters:
            gaussians.use_specular = True

        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, iteration=iteration, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        ref_w = render_pkg["ref_w"]
        out_w = render_pkg["out_w"]
        
        gt_image = viewpoint_cam.original_image.cuda()
            
        loss = 0.0
        
        pbr_rgb = None
  
        if iteration > args.warmup_iters:
            pbr_rgb = render_pkg["pbr_rgb"]
            Ll1 = l1_loss(pbr_rgb, gt_image)
            loss_pbr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pbr_rgb, gt_image))
            loss += loss_pbr
            
            Ll1 = l1_loss(image*out_w, gt_image*out_w)
            loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image*out_w, gt_image*out_w))
            loss += loss_rgb
            
            density_loss = entropy_loss(gaussians.get_opacity[visibility_filter])
            loss += density_loss * 0.001
            
        else:
            Ll1 = l1_loss(image, gt_image)
            loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss += loss_rgb
        

        lambda_normal = 0.05 
        lambda_dist = 0.0 
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        loss = loss + dist_loss + normal_loss

  
        total_loss = loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                eval_lpips = dump_images = iteration == 31_000
                eval(iteration, l1_loss, scene, render_fun, pipe, bg,  dump_images=dump_images, eval_lpips=eval_lpips, ITER=-1, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ )
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("/scratch/sfu/2d-gaussian-splatting/output", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def eval(iteration, l1_loss, scene, renderFunc, pipe, bg, dump_images=False, eval_lpips=False, ITER=-1, ENV_CENTER=None, ENV_RADIUS=None, XYZ=None):
    
    scene.gaussians.eval()
    # Report test and samples of training set
    torch.cuda.empty_cache()
    test_cameras = sorted(scene.getTestCameras(), key=lambda x: x.colmap_id)
    train_cameras = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
    validation_configs = [{'name': 'test', 'cameras' : test_cameras}, {'name': 'train', 'cameras' : train_cameras}]
    os.makedirs(os.path.join(args.model_path, 'render_test'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'render_train'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'roughness'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'normals'), exist_ok=True)
    
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                dump_images = config['name'] == 'test' 
                render_pkg = renderFunc(viewpoint, scene.gaussians, iteration=-1, *(pipe, bg), ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, dump_images=dump_images, idx=idx, config_name=config['name'])
                image = (render_pkg["pbr_rgb"]).clamp(0., 1.)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                
                if eval_lpips:
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
            
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            if eval_lpips:
                lpips_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            write(f'ITER={iteration},{config["name"]}, {psnr_test}, {ssim_test}, {lpips_test}, {scene.gaussians._xyz.shape[0]}')
            
    torch.cuda.empty_cache()
    scene.gaussians.train()
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default = None)
    parser.add_argument('--map_res', type=int, default=48)
    parser.add_argument('--num_sites', type=int, default=2048)
    parser.add_argument('--num_probes', type=int, default=1024)
    parser.add_argument('--warmup_iters', type=int, default=700)
    args = parser.parse_args(sys.argv[1:])

    
    path = args.model_path
    
    args.test_iterations = [1000*i for i in range(1, 31)]
    
    args.eval = True

    if args.scene == 'sedan':
        args.xyz_axis = [2.0, 1.0, 0.0]
        args.images = 'images_8'
        args.env_scope_center = [-0.032, 0.808, 0.751]
        args.env_scope_radius = 2.138 
        args.init_until_iter = 700

    elif args.scene == 'toycar':
        args.xyz_axis = [0.0, 2.0, 1.0]
        args.images = 'images_4'

        args.env_scope_center = [0.486, 1.108, 3.72] 
        args.env_scope_radius = 2.507


    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\nTraining complete.")
