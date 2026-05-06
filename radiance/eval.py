#!/usr/bin/env python3
"""
Minimal evaluation script.

Usage
-----
python eval.py -m <model_path> -s <source_path> \
               [--images images_4] [--iteration -1] \
               [--split test|train|all] [--save_images] \
               [--benchmark] [--warmup 50]

-m / --model_path   trained output folder (contains cfg_args + point_cloud/)
-s / --source_path  original dataset folder  (overrides cfg_args value)
--images            image sub-folder name    (overrides cfg_args value)
--iteration         which saved PLY to load  (-1 = latest, default)
--split             test | train | all       (default: test)
--save_images       write rendered PNGs to <model_path>/eval_images/
--benchmark         measure render time and FPS after quality metrics
--warmup            number of warmup frames before timing (default: 50)
"""

import os
import csv
import time
import numpy as np
import torch
from argparse import ArgumentParser, Namespace

from fused_ssim import fused_ssim
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.beta_model import BetaModel
from utils.camera_utils import cameraList_from_camInfos
from utils.image_utils import psnr
from lpipsPyTorch import lpips


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_cfg_args(model_path: str) -> Namespace:
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return Namespace()
    with open(cfg_path) as f:
        return eval(f.read())          # Namespace(...) string written by train.py


def _cam_args(resolution: int = -1, data_device: str = "cuda") -> Namespace:
    return Namespace(resolution=resolution, data_device=data_device)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main() -> None:
    parser = ArgumentParser(description="Evaluate a trained BetaModel checkpoint")
    parser.add_argument("-m", "--model_path",  required=True)
    parser.add_argument("-s", "--source_path", default=None)
    parser.add_argument("--images",            default=None,
                        help="Image sub-folder override (e.g. images_4)")
    parser.add_argument("--iteration",         type=int, default=-1,
                        help="Saved iteration to load (-1 = latest)")
    parser.add_argument("--split",             default="test",
                        choices=["test", "train", "all"])
    parser.add_argument("--save_images",       action="store_true")
    parser.add_argument("--benchmark",         action="store_true",
                        help="Measure render time / FPS after quality metrics")
    parser.add_argument("--warmup",            type=int, default=100,
                        help="Warmup frames before timing (default: 100)")
    parser.add_argument("--color_rep",         default=None,
                        choices=["voronoi", "sh", "sb", "sgs"],
                        help="Override color representation from cfg_args")
    parser.add_argument("--num_lobes",         type=int, default=None,
                        help="Override num_lobes from cfg_args")
    
    parser.add_argument("--white_background", action="store_true")
    cli = parser.parse_args()

    # ── load training config, then apply CLI overrides ──
    cfg = _read_cfg_args(cli.model_path)

    source_path      = cli.source_path or getattr(cfg, "source_path", None)
    images           = cli.images    
    color_rep        = cli.color_rep   or getattr(cfg, "color_rep",  "voronoi")
    num_lobes        = cli.num_lobes   or getattr(cfg, "num_lobes",  7)
    white_background = cli.white_background or getattr(cfg, "white_background", False)

    assert source_path, "Provide -s <source_path> (or ensure cfg_args has source_path)"
    source_path = os.path.abspath(source_path)

    # ── find PLY ──
    pc_root = os.path.join(cli.model_path, "point_cloud")
    if cli.iteration == -1:
        iteration = searchForMaxIteration(pc_root)
    else:
        iteration = cli.iteration

    ply_path = os.path.join(pc_root, f"iteration_{iteration}", "point_cloud.ply")
    assert os.path.exists(ply_path), f"PLY not found: {ply_path}"
    print(f"  PLY  : {ply_path}")
    print(f"  scene: {source_path}")

    # ── load cameras ──
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            source_path, images, eval=True, init_type="sfm"
        )
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](
            source_path, white_background, eval=True
        )
    else:
        raise ValueError(f"Cannot determine scene type from: {source_path}")

    cam_args = _cam_args()
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, cam_args)
    test_cameras  = cameraList_from_camInfos(scene_info.test_cameras,  1.0, cam_args)

    # ── load model ──
    beta_model = BetaModel(color_rep, num_lobes=num_lobes)
    beta_model.load_ply(ply_path)
    print(f"  loaded model with {beta_model._xyz.shape[0]} primitives, "
          f"color_rep {color_rep}, num_lobes {num_lobes}")
    bg = [1., 1., 1.] if white_background else [0., 0., 0.]
    beta_model.background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    beta_model.eval()
    
    # ── select splits ──
    splits: dict = {}
    if cli.split in ("test",  "all"):
        splits["test"]  = test_cameras
    if cli.split in ("train", "all"):
        splits["train"] = train_cameras

    # ── evaluate ──
    csv_path  = os.path.join(cli.model_path, "eval_results.csv")
    new_file  = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if new_file:
            writer.writerow(["iteration", "split", "psnr", "ssim", "lpips"])

        for split_name, cameras in splits.items():
            if not cameras:
                print(f"  [{split_name}] no cameras, skipping.")
                continue

            img_dir = None
            if cli.save_images:
                img_dir = os.path.join(
                    cli.model_path, "eval_images", f"iter_{iteration}", split_name
                )
                os.makedirs(img_dir, exist_ok=True)

            psnr_sum = ssim_sum = lpips_sum = 0.0

            for cam in cameras:
                render_pkg = beta_model.render(cam)
                img = torch.clamp(render_pkg["render"],          0., 1.)
                gt  = torch.clamp(cam.original_image.cuda(),     0., 1.)

                psnr_sum  += psnr(img, gt).mean().item()
                ssim_sum  += fused_ssim(img.unsqueeze(0), gt.unsqueeze(0)).item()
                lpips_sum += lpips(img * 2. - 1., gt * 2. - 1., net_type="vgg").item()

                if img_dir is not None:
                    from PIL import Image as PILImage
                    np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    PILImage.fromarray(np_img).save(
                        os.path.join(img_dir, f"{cam.image_name}.png")
                    )

            n = len(cameras)
            psnr_val  = psnr_sum  / n
            ssim_val  = ssim_sum  / n
            lpips_val = lpips_sum / n

            print(f"  [iter {iteration}] {split_name:5s}:  "
                  f"PSNR {psnr_val:.2f}   SSIM {ssim_val:.4f}   LPIPS {lpips_val:.4f}")

            writer.writerow([
                iteration, split_name,
                f"{psnr_val:.4f}", f"{ssim_val:.4f}", f"{lpips_val:.4f}",
            ])

    print(f"\n  results → {csv_path}")
    if cli.save_images:
        print(f"  images  → {cli.model_path}/eval_images/iter_{iteration}/")

    # ── benchmark ──
    if cli.benchmark:
        bench_cameras = splits.get("test") or splits.get("train") or []
        if not bench_cameras:
            print("  [benchmark] no cameras available, skipping.")
        else:
            print(f"\n  [benchmark] warmup ({cli.warmup} frames)…", flush=True)
            warmup_pool = bench_cameras * (cli.warmup // len(bench_cameras) + 1)
            for cam in warmup_pool[: cli.warmup]:
                _ = beta_model.render(cam)
            torch.cuda.synchronize()

            print(f"  [benchmark] timing {len(bench_cameras)} frames…", flush=True)
            t0 = time.perf_counter()
            for cam in bench_cameras:
                _ = beta_model.render(cam)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            n_bench = len(bench_cameras)
            fps = n_bench / elapsed
            ms_per_frame = elapsed / n_bench * 1000.0
            print(f"  [benchmark] {n_bench} frames  |  "
                  f"{ms_per_frame:.1f} ms/frame  |  {fps:.1f} FPS")

            bench_csv = os.path.join(cli.model_path, "benchmark_results.csv")
            bench_new = not os.path.exists(bench_csv)
            with open(bench_csv, "a", newline="") as f:
                w = csv.writer(f)
                if bench_new:
                    w.writerow(["iteration", "n_frames", "ms_per_frame", "fps"])
                w.writerow([iteration, n_bench,
                            f"{ms_per_frame:.2f}", f"{fps:.2f}"])
            print(f"  benchmark  → {bench_csv}")


if __name__ == "__main__":
    main()
