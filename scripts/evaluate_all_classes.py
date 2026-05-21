#!/usr/bin/env python3
"""Evaluate all 21 UC Merced classes: edge density, bicubic, RCAN ranking.

Usage:
    python scripts/evaluate_all_classes.py --step edge_density
    python scripts/evaluate_all_classes.py --step bicubic
    python scripts/evaluate_all_classes.py --step ranking --rcan_csv data_experiments/ucmerced_all_classes/results/rcan_x4/metrics.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_metrics, calculate_metrics_standard


def get_class_from_name(name):
    """Extract class name from image filename like 'airplane00.png'."""
    stem = Path(name).stem
    # Remove trailing digits
    class_name = stem.rstrip("0123456789")
    return class_name


def compute_edge_density(img_path):
    """Compute edge density using Canny."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.count_nonzero(edges) / edges.size


def step_edge_density(hr_dir, output_dir):
    """Compute edge density for all HR images."""
    hr_dir = Path(hr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(hr_dir.glob("*.png"))
    print(f"Computing edge density for {len(images)} images...")

    records = []
    class_data = defaultdict(list)

    for i, img_path in enumerate(images):
        ed = compute_edge_density(img_path)
        class_name = get_class_from_name(img_path.name)
        records.append({
            "image": img_path.name,
            "class": class_name,
            "edge_density": ed,
        })
        class_data[class_name].append(ed)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(images)}")

    # Save per-image edge density
    csv_path = output_dir / "edge_density_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class", "edge_density"])
        writer.writeheader()
        writer.writerows(records)

    # Per-class stats
    class_stats = []
    for cls in sorted(class_data.keys()):
        vals = class_data[cls]
        class_stats.append({
            "class": cls,
            "num_images": len(vals),
            "edge_density_mean": np.mean(vals),
            "edge_density_median": np.median(vals),
            "edge_density_max": np.max(vals),
            "edge_density_min": np.min(vals),
        })

    csv_class_path = output_dir / "edge_density_by_class.csv"
    with open(csv_class_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class", "num_images", "edge_density_mean",
            "edge_density_median", "edge_density_max", "edge_density_min"
        ])
        writer.writeheader()
        writer.writerows(class_stats)

    # Print summary
    all_eds = [r["edge_density"] for r in records]
    print(f"\nEdge Density Summary:")
    print(f"  Total images: {len(records)}")
    print(f"  Overall mean: {np.mean(all_eds):.6f}")
    print(f"\n  Per class:")
    for s in sorted(class_stats, key=lambda x: -x["edge_density_mean"]):
        print(f"    {s['class']:25s} mean={s['edge_density_mean']:.6f} "
              f"med={s['edge_density_median']:.6f}")

    # Top/Bottom 10
    sorted_records = sorted(records, key=lambda x: -x["edge_density"])
    print(f"\n  Top 10 edge density:")
    for r in sorted_records[:10]:
        print(f"    {r['image']:30s} {r['edge_density']:.6f}")
    print(f"  Bottom 10 edge density:")
    for r in sorted_records[-10:]:
        print(f"    {r['image']:30s} {r['edge_density']:.6f}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {csv_class_path}")


def step_bicubic(hr_dir, lr_dir, output_dir, scale=4):
    """Evaluate Bicubic x4 with per-class stats."""
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(lr_dir.glob("*.png"))
    print(f"Evaluating Bicubic x{scale} on {len(images)} images...")

    results = []
    class_results = defaultdict(list)

    for i, lr_path in enumerate(images):
        name = lr_path.name
        hr_path = hr_dir / name
        if not hr_path.exists():
            print(f"  WARNING: HR not found for {name}, skipping")
            continue

        hr = np.array(Image.open(hr_path).convert("RGB"))
        lr = np.array(Image.open(lr_path).convert("RGB"))
        h, w = hr.shape[:2]

        # Bicubic upsample
        sr = np.array(Image.fromarray(lr).resize((w, h), Image.BICUBIC))

        # RGB metrics
        rgb_m = calculate_metrics(hr, sr)
        # Y+crop metrics
        y_m = calculate_metrics_standard(hr, sr, scale=scale)

        class_name = get_class_from_name(name)
        result = {
            "image": name,
            "class": class_name,
            "rgb_psnr": rgb_m["psnr"],
            "rgb_ssim": rgb_m["ssim"],
            "y_psnr": y_m["psnr"],
            "y_ssim": y_m["ssim"],
        }
        results.append(result)
        class_results[class_name].append(result)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(images)}")

    # Save per-image
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "class", "rgb_psnr", "rgb_ssim", "y_psnr", "y_ssim"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Per-class summary
    class_stats = []
    for cls in sorted(class_results.keys()):
        vals = class_results[cls]
        class_stats.append({
            "class": cls,
            "num_images": len(vals),
            "rgb_psnr_mean": np.mean([v["rgb_psnr"] for v in vals]),
            "rgb_ssim_mean": np.mean([v["rgb_ssim"] for v in vals]),
            "y_psnr_mean": np.mean([v["y_psnr"] for v in vals]),
            "y_ssim_mean": np.mean([v["y_ssim"] for v in vals]),
        })

    csv_class_path = output_dir.parent.parent / "report_assets" / "tables" / "ucmerced_all_classes" / "bicubic_x4_by_class.csv"
    csv_class_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_class_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class", "num_images", "rgb_psnr_mean", "rgb_ssim_mean",
            "y_psnr_mean", "y_ssim_mean"
        ])
        writer.writeheader()
        writer.writerows(class_stats)

    # Overall summary
    all_rgb_psnr = [r["rgb_psnr"] for r in results]
    all_y_psnr = [r["y_psnr"] for r in results]
    summary_lines = [
        "Bicubic x4 Evaluation - UC Merced All Classes",
        "=" * 50,
        f"Total images: {len(results)}",
        f"Overall RGB PSNR: {np.mean(all_rgb_psnr):.2f} dB",
        f"Overall Y+crop PSNR: {np.mean(all_y_psnr):.2f} dB",
        "",
        "Per class (sorted by Y+crop PSNR):",
    ]
    for s in sorted(class_stats, key=lambda x: -x["y_psnr_mean"]):
        summary_lines.append(
            f"  {s['class']:25s} Y={s['y_psnr_mean']:.2f}  RGB={s['rgb_psnr_mean']:.2f}"
        )
    summary = "\n".join(summary_lines)

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    summary_report_path = output_dir.parent.parent / "report_assets" / "tables" / "ucmerced_all_classes" / "bicubic_x4_summary.txt"
    with open(summary_report_path, "w") as f:
        f.write(summary)

    print(f"\n{summary}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {csv_class_path}")
    print(f"Saved: {summary_path}")


def step_ranking(rcan_csv, bicubic_csv, edge_csv, output_dir):
    """Generate ranking table combining RCAN, Bicubic, and edge density."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load edge density by class
    edge_by_class = {}
    if Path(edge_csv).exists():
        with open(edge_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                edge_by_class[row["class"]] = float(row["edge_density_mean"])

    # Load bicubic by image
    bicubic_by_image = {}
    bicubic_by_class = defaultdict(list)
    if Path(bicubic_csv).exists():
        with open(bicubic_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bicubic_by_image[row["image"]] = {
                    "y_psnr": float(row["y_psnr"]),
                    "rgb_psnr": float(row["rgb_psnr"]),
                }
                bicubic_by_class[row["class"]].append(float(row["y_psnr"]))

    # Load RCAN by image
    rcan_by_image = {}
    with open(rcan_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rcan_by_image[row["image"]] = {
                "y_psnr": float(row.get("y_psnr", row.get("psnr", 0))),
                "y_ssim": float(row.get("y_ssim", row.get("ssim", 0))),
                "rgb_psnr": float(row.get("rgb_psnr", row.get("psnr", 0))),
                "rgb_ssim": float(row.get("rgb_ssim", row.get("ssim", 0))),
            }

    # Group by class
    rcan_by_class = defaultdict(list)
    for name, m in rcan_by_image.items():
        cls = get_class_from_name(name)
        rcan_by_class[cls].append(m)

    # Build ranking
    ranking = []
    for cls in sorted(rcan_by_class.keys()):
        rcan_vals = rcan_by_class[cls]
        bicubic_vals = bicubic_by_class.get(cls, [])

        rcan_y = np.mean([v["y_psnr"] for v in rcan_vals])
        rcan_y_ssim = np.mean([v["y_ssim"] for v in rcan_vals])
        rcan_rgb = np.mean([v["rgb_psnr"] for v in rcan_vals])
        rcan_rgb_ssim = np.mean([v["rgb_ssim"] for v in rcan_vals])
        bicubic_y = np.mean(bicubic_vals) if bicubic_vals else 0

        ranking.append({
            "class": cls,
            "num_images": len(rcan_vals),
            "edge_density_mean": edge_by_class.get(cls, 0),
            "bicubic_y_psnr": bicubic_y,
            "rcan_y_psnr": rcan_y,
            "gain_vs_bicubic": rcan_y - bicubic_y,
            "rcan_y_ssim": rcan_y_ssim,
            "rcan_rgb_psnr": rcan_rgb,
            "rcan_rgb_ssim": rcan_rgb_ssim,
        })

    # Sort by RCAN Y+crop PSNR descending
    ranking.sort(key=lambda x: -x["rcan_y_psnr"])

    # Add rank
    for i, r in enumerate(ranking):
        r["rank"] = i + 1

    # Save ranking CSV
    fields = ["rank", "class", "num_images", "edge_density_mean",
              "bicubic_y_psnr", "rcan_y_psnr", "gain_vs_bicubic",
              "rcan_y_ssim", "rcan_rgb_psnr", "rcan_rgb_ssim"]
    csv_path = output_dir / "rcan_x4_by_class_rank.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranking)

    # Generate summary
    lines = [
        "UC Merced All Classes - RCAN x4 Ranking",
        "=" * 60,
        "",
        "Full Ranking (by Y+crop PSNR):",
    ]
    for r in ranking:
        lines.append(
            f"  #{r['rank']:2d} {r['class']:25s} "
            f"Y={r['rcan_y_psnr']:.2f}  RGB={r['rcan_rgb_psnr']:.2f}  "
            f"gain={r['gain_vs_bicubic']:+.2f}  edge={r['edge_density_mean']:.4f}"
        )

    top5 = [r["class"] for r in ranking[:5]]
    top8 = [r["class"] for r in ranking[:8]]
    top10 = [r["class"] for r in ranking[:10]]
    below30 = [r["class"] for r in ranking if r["rcan_y_psnr"] < 30]
    highest = ranking[0]
    lowest = ranking[-1]

    lines.extend([
        "",
        f"Top 5: {', '.join(top5)}",
        f"Top 8: {', '.join(top8)}",
        f"Top 10: {', '.join(top10)}",
        f"Below 30dB: {', '.join(below30) if below30 else 'None'}",
        f"Highest: {highest['class']} ({highest['rcan_y_psnr']:.2f} dB)",
        f"Lowest: {lowest['class']} ({lowest['rcan_y_psnr']:.2f} dB)",
    ])

    summary = "\n".join(lines)
    summary_path = output_dir / "rcan_x4_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(summary)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True,
                        choices=["edge_density", "bicubic", "ranking"])
    parser.add_argument("--hr_dir", type=str,
                        default="data_experiments/ucmerced_all_classes/HR")
    parser.add_argument("--lr_dir", type=str,
                        default="data_experiments/ucmerced_all_classes/LR_x4")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--rcan_csv", type=str, default=None)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    report_dir = Path("report_assets/tables/ucmerced_all_classes")
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.step == "edge_density":
        output = args.output_dir or str(report_dir)
        step_edge_density(args.hr_dir, output)

    elif args.step == "bicubic":
        output = args.output_dir or "data_experiments/ucmerced_all_classes/results/bicubic_x4"
        step_bicubic(args.hr_dir, args.lr_dir, output, args.scale)

    elif args.step == "ranking":
        rcan_csv = args.rcan_csv or "data_experiments/ucmerced_all_classes/results/rcan_x4/metrics.csv"
        bicubic_csv = "data_experiments/ucmerced_all_classes/results/bicubic_x4/metrics.csv"
        edge_csv = str(report_dir / "edge_density_by_class.csv")
        output = args.output_dir or str(report_dir)
        step_ranking(rcan_csv, bicubic_csv, edge_csv, output)


if __name__ == "__main__":
    main()
