import sys
import torch
import torchvision
import numpy as np
import PIL
import cv2
import skimage
import matplotlib
import yaml

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("cuDNN:", torch.backends.cudnn.version())
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("GPU tensor test:", y.mean().item())
else:
    raise RuntimeError("CUDA is not available in PyTorch.")

print("NumPy:", np.__version__)
print("Pillow:", PIL.__version__)
print("OpenCV:", cv2.__version__)
print("skimage:", skimage.__version__)
print("matplotlib:", matplotlib.__version__)
print("PyYAML:", yaml.__version__)
print("environment check passed")
