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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.beta_model import BetaModel
from arguments import ModelParams
from utils.camera_utils import (
    cameraList_from_camInfos,
    camera_to_JSON,
)
import torch
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from fused_ssim import fused_ssim
from tqdm import tqdm
import time


class Scene:
    beta_model: BetaModel

    def __init__(
        self,
        args: ModelParams,
        beta_model: BetaModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.beta_model = beta_model
        self.best_psnr = 0

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.val_cameras = {}
        self.test_cameras = {}
        self.other_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, init_type=args.init_type
        )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
        )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
        if scene_info.other_cameras:
            print("Loading Other Cameras")
            for i, cameras in enumerate(scene_info.other_cameras):
                self.other_cameras[f'zoom{i}'] = cameraList_from_camInfos(cameras, resolution_scale, args, cam_only=True)

        if self.loaded_iter:
            self.beta_model.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.beta_model.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.beta_model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getOtherCameras(self, nameset):
        return self.other_cameras[nameset]


    @torch.no_grad()
    def save_best_model(self):
        psnr_test = 0.0
        test_view_stack = self.getTestCameras()
        for idx, viewpoint in enumerate(test_view_stack):
            image = torch.clamp(self.beta_model.render(viewpoint)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean()
        psnr_test /= len(test_view_stack)
        if psnr_test > self.best_psnr:
            self.save("best")
            self.best_psnr = psnr_test
            return True
        else:
            return False


    @torch.no_grad()
    def eval(self, name_set, dump_images=False, eval_lpips=False, iteration=None):
        
        if name_set not in ['test_torch']:
            self.beta_model.eval()
        dir_path = os.path.join(self.model_path, f'render_{name_set}')
        if iteration is not None and dump_images:
            dir_path += f'iter={iteration}'
            
        os.makedirs(dir_path, exist_ok=True)
            
        torch.cuda.empty_cache()
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        fps_test = 0.0
        test_view_stack = None
        if name_set == 'train':
            test_view_stack = self.getTrainCameras()
        elif name_set in ['test', 'test_torch']:
            test_view_stack = self.getTestCameras()
        elif 'val' in name_set:
            test_view_stack = self.val_cameras
            
        test_view_stack = sorted(test_view_stack, key=lambda x: x.colmap_id)
            
        for idx, viewpoint in tqdm(enumerate(test_view_stack)):
            start = time.time()
            image = torch.clamp(self.beta_model.render(viewpoint)["render"], 0.0, 1.0)
            end = time.time() - start
            if dump_images:
                np_img = np.transpose(image.cpu().numpy(), (1, 2, 0))
                np_img = (np_img * 255).astype(np.uint8)
                img_ = Image.fromarray(np_img)
                img_.save(os.path.join(dir_path, f'{idx}.png'))

            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().item()
            ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            fps_test += 1 / end
            if eval_lpips:
                lpips_test += lpips(image*2-1, gt_image*2-1, net_type="vgg").mean().item()

        psnr_test /= len(test_view_stack)
        ssim_test /= len(test_view_stack)
        lpips_test /= len(test_view_stack)
        fps_test /= len(test_view_stack)

        result = psnr_test, ssim_test, lpips_test, fps_test
        self.beta_model.train()
        
        return result
