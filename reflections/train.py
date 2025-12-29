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
import os
import torch
from utils.loss_utils import entropy_loss, l1_loss, ssim
from gaussian_renderer import network_gui, render_df
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
import time 


TENSORBOARD_FOUND = False


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input + 1e-10) + (1 - target) * torch.log(1 - input + 1e-10)).mean()


def write(string):
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
        print(string, file=f)
        
        
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0

    render_fun = render_df 
    
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(num_probes=args.num_probes, num_sites=args.num_sites, map_res=args.map_res)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint or args.eval_only:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    gaussians.model_path = args.model_path


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    

    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    gaussians.use_specular = False
    gaussians.train()
    start = time.time()
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
        
        bg = torch.rand((3), device="cuda") if args.rand_bg else torch.zeros((3), device='cuda')
        
        if args.eval_only:
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            testing_iterations, scene, render_fun, (pipe, bg), eval_lpips=False, dump_images=False)
            return
        
        if iteration == 1 or iteration % 501 == 0:
            gaussians.compute_top_idx()
        
        if iteration == args.warmup_iters: # first warmup_iters iter diffuse only
            gaussians.use_specular = True
        
        if iteration == 28_000: #disable learning of positions and sites 
            for param in gaussians.optimizer.param_groups:
                if param['name'] in ['sites_lp', 'positions_lp']: 
                    param['lr'] = 0


        gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera (same as ref-gs)
        data_idx = np.random.randint(len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack[data_idx]     
        
        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, iteration, dump_images=False)
        
        viewspace_point_tensor, visibility_filter, radii =  render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        num_channels = gt_image.shape[0]
        if num_channels == 4:
            gt_image = gt_image[:3,...] * gt_image[3:,...] + (1-gt_image[3:,...]) * bg[:, None, None]

        loss = 0.0
        pbr_rgb = None
        Ll1 = 0.0
       
        pbr_rgb = render_pkg["render"] * render_pkg["rend_alpha"] + (1-render_pkg["rend_alpha"]) * bg[:, None, None]
        Ll1 = l1_loss(pbr_rgb, gt_image)
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pbr_rgb, gt_image))
        loss += loss_pbr
        
        if iteration < 3000 and num_channels == 4:
            gt_mask = viewpoint_cam.original_image.cuda()[3:,...]
            alpha_loss = binary_cross_entropy(render_pkg["rend_alpha"], gt_mask)
            loss += alpha_loss
        
        if iteration % 50 == 0:
            torchvision.utils.save_image((render_pkg['rend_normal'] + 1)/2, 'normals.png')
        
        lambda_normal = 0.05 if iteration < 700 else 0.05
        

        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        total_loss = loss + normal_loss 
        
        
        total_loss.backward()

        iter_end.record()


        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.0 
            ema_normal_for_log = 0.0 if not gaussians.use_specular else 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


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

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
            dump_images = eval_lpips = iteration == args.iterations
            
            if iteration == args.iterations:
                end = time.time()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_fun, (pipe, bg),
                            eval_lpips=eval_lpips, dump_images=dump_images)
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
                    if gaussians.color_representation == 'voronoi':
                        gaussians.update_sites_mask()
                        gaussians.compute_top_idx()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
            
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render_df(custom_cam, gaussians, pipe, bg, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None
                    
    train_time = end - start
    write(f"train_time, {train_time}")
    

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
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, 
                    renderArgs, dump_images=False, eval_lpips=False):
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar('iter_time', elapsed, iteration)
    #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
    os.makedirs(os.path.join(args.model_path, 'render_test'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'render_train'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'gt_test'), exist_ok=True)

    # Report test and samples of training set
    if iteration in testing_iterations or args.eval_only:
        
        bg = renderArgs[1]
        
        scene.gaussians.eval()
        
        if scene.gaussians.color_representation == 'voronoi':
                to_zero = 0 if scene.gaussians._mask is None else (scene.gaussians._mask == 0).sum() / scene.gaussians._mask.numel()
        else:
            to_zero = 0
        torch.cuda.empty_cache()
        test_cameras = sorted(scene.getTestCameras(), key=lambda x: x.colmap_id)
        train_cameras = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
        validation_configs = [{'name': 'test', 'cameras' : test_cameras}, 
                              {'name': 'train', 'cameras' : train_cameras}]
        
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for config in validation_configs:
            
            if config['name'] == 'train':
                continue

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                fps_list = []
                dump_images = config['name'] == 'test' and iteration == 30_000
                for idx, viewpoint in enumerate(config['cameras']):
                    torch.cuda.synchronize()
                    start.record()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, dump_images=dump_images, config_name=config['name'], idx=idx)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    image = (image * render_pkg["rend_alpha"] + (1-render_pkg["rend_alpha"]) * bg[:, None, None]).clamp(0., 1.)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end)  
                    fps = 1000 / elapsed_ms
                    fps_list.append(fps)
                    
                    rend_normal  = render_pkg['rend_normal']

                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if gt_image.shape[0] == 4:
                        gt_image = gt_image[:3,...] * gt_image[3:,...] + (1-gt_image[3:,...]) * bg[:, None, None]
                        
                    if dump_images:
                        torchvision.utils.save_image(gt_image, f"{args.model_path}/gt_test/{idx}.png")
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    if eval_lpips:
                        #lpips expects inputs in [-1, 1] but previous baselines did not do that, so we kept it like this for fair comparison
                        lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                fps_test = np.mean(fps_list[2:])
                if eval_lpips:
                    lpips_test /= len(config['cameras'])
                #if config['name'] == 'test':
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                write(f'ITER={iteration},{config["name"]},{psnr_test},{ssim_test},{lpips_test},{scene.gaussians._xyz.shape[0]},{fps_test}')
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

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
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default = None)
    parser.add_argument("--config", type=str, default = None)
    parser.add_argument('--map_res', type=int, default=48)
    parser.add_argument('--num_sites', type=int, default=2048)
    parser.add_argument('--num_probes', type=int, default=128)
    parser.add_argument('--warmup_iters', type=int, default=700)
    parser.add_argument("--rand_bg", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    
    
    path = args.model_path
    
    args.iterations = 30000
    
    args.test_iterations = [1000*i for i in range(1, 31)]

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\nTraining complete.")
