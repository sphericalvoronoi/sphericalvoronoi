#!/usr/bin/env python3
"""
Summarise eval_results.csv files across all benchmark datasets.

Usage
-----
python summary.py [--output_root <dir>] [--split test] [--iteration -1]
                  [--save [path]]

--output_root   base directory that contains per-scene output folders
                (default: <repo>/output)
--split         which split row to read (default: test)
--iteration     iteration to report; -1 means the highest available (default: -1)
--save          write a CSV summary (default path: <output_root>/summary.csv)
"""

import os
import csv
import argparse
import sys

# ---------------------------------------------------------------------------
# Dataset registry  (MipNeRF-360 = indoor + outdoor together)
# ---------------------------------------------------------------------------

DATASETS = {
    "MipNeRF-360": [
        "bonsai", "kitchen", "room", "counter",
        "bicycle", "garden", "flowers", "stump", "treehill",
    ],
    "Tanks & Temples": [
        "train", "truck",
    ],
    "Deep Blending": [
        "drjohnson", "playroom",
    ],
    "NeRF Synthetic": [
        "lego", "mic", "ship", "chair", "ficus", "hotdog", "materials", "drums",
    ],
}

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def load_eval_csv(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def best_row(rows: list[dict], split: str, iteration: int) -> dict | None:
    rows = [r for r in rows if r.get("split") == split]
    if not rows:
        return None
    if iteration == -1:
        rows.sort(key=lambda r: int(r["iteration"]))
        return rows[-1]
    matches = [r for r in rows if int(r["iteration"]) == iteration]
    return matches[-1] if matches else None


def _f(val, fmt) -> str:
    try:
        return format(float(val), fmt) if val not in (None, "", "-") else "-"
    except (ValueError, TypeError):
        return "-"


# ---------------------------------------------------------------------------
# Pretty-print table (stdout only)
# ---------------------------------------------------------------------------

def print_table(results: dict, args) -> None:
    """results: {dataset_name: {scene: row_dict | None}}"""
    col_headers = ["Scene", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "Iter", "#Gauss"]
    col_w = [18, 9, 9, 9, 8, 10]
    inner_w = sum(col_w) + len(col_w) - 1

    def emit(line=""):
        print(line)

    def emit_row(cells):
        emit("│" + "│".join(c.center(w) for c, w in zip(cells, col_w)) + "│")

    emit("┌" + "┬".join("─" * w for w in col_w) + "┐")
    emit("│" + "│".join(h.center(w) for h, w in zip(col_headers, col_w)) + "│")

    grand_psnr = grand_ssim = grand_lpips = 0.0
    grand_n = grand_lpips_n = 0

    first = True
    for dataset_name, scene_rows in results.items():
        emit("├" + ("┬" if first else "─").join("─" * w for w in col_w) + "┤" if first
             else "├" + "─" * inner_w + "┤")
        first = False
        emit("│" + f" {dataset_name} ".center(inner_w) + "│")
        emit("├" + "┼".join("─" * w for w in col_w) + "┤")

        ds_psnr = ds_ssim = ds_lpips = 0.0
        ds_n = ds_lpips_n = 0
        missing = []

        for scene, row in scene_rows.items():
            if row is None:
                missing.append(scene)
                emit_row([scene, "-", "-", "-", "-", "-"])
                continue
            gauss_s = row.get("num_gaussians", "-")
            if gauss_s not in ("", "-", None):
                try:
                    gauss_s = f"{int(gauss_s):,}"
                except ValueError:
                    pass
            emit_row([
                scene,
                _f(row.get("psnr"),  ".2f"),
                _f(row.get("ssim"),  ".4f"),
                _f(row.get("lpips"), ".4f"),
                row.get("iteration", "-"),
                gauss_s,
            ])
            try:
                ds_psnr += float(row["psnr"]); ds_ssim += float(row["ssim"]); ds_n += 1
                if row.get("lpips", "") not in ("", None):
                    ds_lpips += float(row["lpips"]); ds_lpips_n += 1
            except (ValueError, KeyError):
                pass

        emit("├" + "┼".join("─" * w for w in col_w) + "┤")
        if ds_n > 0:
            avg_lpips = ds_lpips / ds_lpips_n if ds_lpips_n else None
            emit_row([
                f"avg ({ds_n}/{len(scene_rows)})",
                format(ds_psnr / ds_n, ".2f"),
                format(ds_ssim / ds_n, ".4f"),
                format(avg_lpips, ".4f") if avg_lpips is not None else "-",
                "", "",
            ])
            grand_psnr += ds_psnr; grand_ssim += ds_ssim; grand_n += ds_n
            grand_lpips += ds_lpips; grand_lpips_n += ds_lpips_n
        else:
            emit_row(["avg", "-", "-", "-", "", ""])

        if missing:
            emit("│" + f" missing: {', '.join(missing)} ".ljust(inner_w) + "│")

    emit("├" + "─" * inner_w + "┤")
    emit("│" + " Overall average ".center(inner_w) + "│")
    emit("├" + "┼".join("─" * w for w in col_w) + "┤")
    if grand_n > 0:
        g_lpips = grand_lpips / grand_lpips_n if grand_lpips_n else None
        emit_row([
            f"all ({grand_n} scenes)",
            format(grand_psnr / grand_n, ".2f"),
            format(grand_ssim / grand_n, ".4f"),
            format(g_lpips, ".4f") if g_lpips is not None else "-",
            "", "",
        ])
    else:
        emit_row(["all", "-", "-", "-", "", ""])
    emit("└" + "┴".join("─" * w for w in col_w) + "┘")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(results: dict, path: str) -> None:
    """
    Writes one row per scene plus one 'avg' row per dataset.
    Columns: dataset, scene, psnr, ssim, lpips, iteration, num_gaussians
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "scene", "psnr", "ssim", "lpips",
                         "iteration", "num_gaussians"])

        for dataset_name, scene_rows in results.items():
            ds_psnr = ds_ssim = ds_lpips = 0.0
            ds_n = ds_lpips_n = 0

            for scene, row in scene_rows.items():
                if row is None:
                    writer.writerow([dataset_name, scene, "", "", "", "", ""])
                    continue
                psnr_v  = _f(row.get("psnr"),  ".4f")
                ssim_v  = _f(row.get("ssim"),  ".4f")
                lpips_v = _f(row.get("lpips"), ".4f")
                writer.writerow([
                    dataset_name, scene,
                    psnr_v, ssim_v, lpips_v,
                    row.get("iteration", ""),
                    row.get("num_gaussians", ""),
                ])
                try:
                    ds_psnr += float(row["psnr"]); ds_ssim += float(row["ssim"]); ds_n += 1
                    if row.get("lpips", "") not in ("", None):
                        ds_lpips += float(row["lpips"]); ds_lpips_n += 1
                except (ValueError, KeyError):
                    pass

            if ds_n > 0:
                avg_lpips = ds_lpips / ds_lpips_n if ds_lpips_n else ""
                writer.writerow([
                    dataset_name, "avg",
                    f"{ds_psnr / ds_n:.4f}",
                    f"{ds_ssim / ds_n:.4f}",
                    f"{avg_lpips:.4f}" if avg_lpips != "" else "",
                    "", "",
                ])
            else:
                writer.writerow([dataset_name, "avg", "", "", "", "", ""])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark summary")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory with per-scene output folders")
    parser.add_argument("--split",     type=str, default="test",
                        choices=["test", "train"])
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration to report (-1 = latest)")
    parser.add_argument("--save", nargs="?", const=True, default=True,
                        metavar="PATH",
                        help="Write CSV summary (default: <output_root>/summary.csv)")
    args = parser.parse_args()

    if args.output_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_root = os.path.join(script_dir, "..", "output")

    # Collect results
    results: dict[str, dict[str, dict | None]] = {}
    for dataset_name, scenes in DATASETS.items():
        results[dataset_name] = {}
        for scene in scenes:
            csv_path = os.path.join(args.output_root, scene, "eval_results.csv")
            rows = load_eval_csv(csv_path)
            results[dataset_name][scene] = best_row(rows, args.split, args.iteration)

    # Pretty table to stdout
    print_table(results, args)

    # CSV to file
    if args.save is not None:
        save_path = (
            args.save if isinstance(args.save, str)
            else os.path.join(args.output_root, "summary.csv")
        )
        write_csv(results, save_path)
        print(f"\nCSV saved to {save_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
