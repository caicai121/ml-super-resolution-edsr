#!/usr/bin/env python3
"""Patch train.py to log residual_alpha if model has it."""

with open("train.py", "r") as f:
    content = f.read()

# Add alpha logging after the epoch print (in the else branch)
old_print = '''        else:
            print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f} | "
                  f"RGB: {val_m['rgb_psnr']:.2f}/{val_m['rgb_ssim']:.4f} | "
                  f"Y+crop: {val_m['y_psnr']:.2f}/{val_m['y_ssim']:.4f} | "
                  f"Best Y: {best_y_psnr:.2f}")'''

new_print = '''        else:
            alpha_str = ""
            if hasattr(model, 'residual_alpha'):
                alpha_str = f" | alpha: {model.residual_alpha.item():.4f}"
            print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f} | "
                  f"RGB: {val_m['rgb_psnr']:.2f}/{val_m['rgb_ssim']:.4f} | "
                  f"Y+crop: {val_m['y_psnr']:.2f}/{val_m['y_ssim']:.4f} | "
                  f"Best Y: {best_y_psnr:.2f}{alpha_str}")'''

if old_print in content:
    content = content.replace(old_print, new_print)
    with open("train.py", "w") as f:
        f.write(content)
    print("Patched train.py to log residual_alpha")
else:
    print("Could not find target print statement in train.py")
    print("Searching for partial match...")
    if "Best Y: {best_y_psnr:.2f}" in content:
        print("Found 'Best Y' string but exact match failed")
    else:
        print("Pattern not found at all")
