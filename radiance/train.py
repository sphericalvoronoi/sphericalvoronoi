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

import os
import torch.nn.functional as F
import torch
from random import randint
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim
import sys
from scene import Scene, BetaModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, ViewerParams, OptimizationParams
from scene.beta_model import build_scaling_rotation
import json


cap_max = {
    "bicycle": 6_000_000,
    "flowers": 3_000_000,
    "garden": 5_000_000,
    "stump": 4_500_000,
    "treehill": 3_500_000,
    "room": 1_500_000,
    "counter": 1_500_000,
    "kitchen": 1_500_000,
    "bonsai": 1_500_000,
    "train": 1_000_000,
    "truck": 2_500_000,
    "drjohnson": 3_500_000,
    "playroom": 2_500_000,
    "lego":  325000,
    "mic": 320000,
    "ship": 360000,
    "materials": 290000,
    "chair": 270000,
    "hotdog": 150000,
    'ficus': 300000,
    'drums': 350000
}


def uniform_spherical_loss(sites):
    v = F.normalize(sites, dim=-1)            
    dots = torch.einsum('nkd,njd->nkj', v, v)  
    K = sites.shape[1]
    eye = torch.eye(K, device=sites.device)[None]
    mask = ~eye.bool()                         
    dots_off = dots[mask.expand_as(dots)]      
    target = -1.0 / (K - 1)
    loss = ((dots_off - target) ** 2).mean()
    return loss


def write(string):
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
        print(string, file=f)


def training(args):
    first_iter = 0
    prepare_output_and_logger(args)
    beta_model = BetaModel(args.color_rep, num_lobes=args.num_lobes)
    scene = Scene(args, beta_model, load_iteration=None)
    beta_model.training_setup(args)
    
    if args.start_checkpoint:
        model_params = torch.load(args.start_checkpoint)
        beta_model.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    beta_model.background = (
            torch.rand((3), device="cuda") if args.random_background else background
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    iteration = first_iter + 1
    if args.cap_max < beta_model._xyz.shape[0]:
        print(
            f"Warning: cap_max ({args.cap_max}) is smaller than the number of points initialized ({beta_model._xyz.shape[0]}). Resetting cap_max to the number of points initialized."
        )
        args.cap_max = beta_model._xyz.shape[0]
    if not args.eval:
        progress_bar = tqdm(
            range(first_iter, args.iterations), desc="Training progress"
        )
    else:
        progress_bar = tqdm(desc="Training progress")

    beta_model.train()
    
    while True and not args.eval_only:
        if iteration > args.iterations:
            break

        iter_start.record()

        xyz_lr = beta_model.update_learning_rate(iteration)
        
        if iteration % 1000 == 0:
            beta_model.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
              
        render_pkg = beta_model.render(viewpoint_cam, mask=None)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (
            1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        )

        
        if args.densify_from_iter < iteration < args.densify_until_iter:
            loss += args.opacity_reg * torch.abs(beta_model.get_opacity).mean()
            loss += args.scale_reg * torch.abs(beta_model.get_scaling).mean()
            
            if beta_model.color_representation == 'voronoi':
                loss += args.l_l1 * beta_model._colors.abs().sum(dim=-1).mean() 
                
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix(
                    {
                        "Iter": iteration,
                        "Loss": f"{ema_loss_for_log:.4f}",
                        "Beta": f"{beta_model._beta.mean().item():.2f}",
                    }
                )
                progress_bar.update(100)
                
            if (
                iteration < args.densify_until_iter
                and iteration > args.densify_from_iter
                and iteration % args.densification_interval == 0
            ):
                dead_mask = (beta_model.get_opacity <= 0.005).squeeze(-1)
                beta_model.relocate_gs(dead_mask=dead_mask)
                beta_model.add_new_gs(cap_max=args.cap_max)

                L = build_scaling_rotation(
                    beta_model.get_scaling, beta_model.get_rotation
                )
                actual_covariance = L @ L.transpose(1, 2)

                noise = (
                    torch.randn_like(beta_model._xyz)
                    * (torch.pow(1 - beta_model.get_opacity, 100))
                    * args.noise_lr
                    * xyz_lr
                )
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                beta_model._xyz.add_(noise)
                
                if beta_model.color_representation == 'voronoi':
                    beta_model.update_sites_mask()

            beta_model.optimizer.step()
            beta_model.optimizer.zero_grad(set_to_none=True)

            if iteration in test_iterations:
                for set in ['test_torch', 'test', 'train' ]:   
                    result = scene.eval(set, iteration=iteration)
                    write(f'ITER={iteration},{set.upper()},{result[0]},{result[1]},{result[2]},{result[3]}')
                    print(f'ITER={iteration}: PSNR {set.upper()} {result[0]}, SSIM {set} {result[1]}, FPS {set} {result[3]}')
                    
            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving beta_model")
                scene.save(iteration)
            
            if iteration in args.checkpoint_iterations:
                beta_model.save(os.path.join(args.model_path, f'ckpts_ITER={iteration}.pth'))

        iteration += 1

    progress_bar.close()

    for set in ['test']: #'train   
        result = scene.eval(set, dump_images=True, eval_lpips=True, iteration=iteration)
        write(f'ITER={iteration},{set.upper()},{result[0]},{result[1]},{result[2]},{result[3]}')
        print(f'ITER={iteration}: PSNR {set.upper()} {result[0]}, SSIM {set} {result[1]}, FPS {set} {result[3]}')


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
        
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ModelParams(parser), OptimizationParams(parser), ViewerParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30_000])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--color_rep", type=str, default='voronoi')
    parser.add_argument("--num_lobes", type=int, default=8)
    parser.add_argument("--eval_only", action='store_true', default=False)
    
    parser.add_argument(
        "--no-compress",
        action="store_false",
        dest="compress",
        help="Disable compression (compression is on by default)",
    )

    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)
            
            
    path = args.model_path
                  
    test_iterations = [1000*i for i in range(1, 31_000)]
    args.checkpoint_iterations = []
    args.cap_max = cap_max[args.scene]
    args.eval = True
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(args)
