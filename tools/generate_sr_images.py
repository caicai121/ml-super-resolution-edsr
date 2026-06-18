#!/usr/bin/env python3
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml

BASE = Path("/root/Code/ml-super-resolution-edsr")
sys.path.insert(0, str(BASE))

from models.rcan import RCAN, MSRRCAN, MSRRCANV2, CascadeMSRRCAN
from utils.dataset import SRDataset
from utils.image_utils import tensor_to_np

def load_model(cfg_path, ckpt_path, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mp = cfg.get("model_params", {})
    model_name = cfg["model"]
    if model_name == "rcan_small":
        model = RCAN(in_channels=3, out_channels=3, num_features=64, num_resgroups=3, num_resblocks=5, reduction=16, scale=cfg["scale"])
    elif model_name in ["msr_rcan", "msr_rcan_mid", "msr_rcan_large"]:
        model = MSRRCANV2(in_channels=3, out_channels=3, num_features=64, num_resgroups=mp.get("n_resgroups", 5), num_resblocks=mp.get("n_resblocks", 5), reduction=16, scale=cfg["scale"])
    elif model_name in ["cascade_msr_rcan", "cascade_msr_rcan_large"]:
        model = CascadeMSRRCAN(in_channels=3, out_channels=3, num_features=64, num_resgroups=8, num_resblocks=8, reduction=16, scale=cfg["scale"], cascade_num_blocks=mp.get("cascade_num_blocks", 10), cascade_mid_channels=64, cascade_residual_scale=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, cfg

def generate_sr_images(model_cfg, ckpt_path, output_dir, target_images, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(model_cfg, ckpt_path, device)
    dataset = SRDataset(hr_dir=BASE / "data_final" / "ucmerced_selected" / "test" / "HR", lr_dir=BASE / "data_final" / "ucmerced_selected" / "test" / "LR_x4", scale=cfg["scale"])
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    generated = []
    with torch.no_grad():
        for batch in loader:
            name = batch["name"][0]
            if name not in target_images:
                continue
            lr = batch["lr"].to(device)
            sr = model(lr)
            sr_np = tensor_to_np(sr[0])
            out_path = output_dir / name
            Image.fromarray((sr_np * 255).astype(np.uint8)).save(out_path)
            generated.append(name)
            print(f"  Saved: {out_path}")
    return generated

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    target_images = {"runway_runway92.png", "tenniscourt_tenniscourt60.png", "airplane_airplane14.png"}
    print("\n=== Cascade-10 ===")
    cascade_gen = generate_sr_images(
        BASE / "configs" / "cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml",
        BASE / "checkpoints" / "cascade_msr_rcan_large_s10_50_cosine_x4" / "best_cascade_msr_rcan_large50_cosine_x4.pth",
        BASE / "report_assets" / "figures" / "final_report" / "sr_images" / "cascade_10",
        target_images, device)
    print(f"Generated: {cascade_gen}")
    print("\nDONE!")
