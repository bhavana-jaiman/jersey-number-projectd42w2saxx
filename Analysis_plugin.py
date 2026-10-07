import os
import torch
from PIL import Image

def plug_save_image(img_tensor, output_dir="output"):
    """Save a (1, 3, 96, 96) image tensor as <i>.jpeg, incrementing i on each call."""
    if not hasattr(plug_save_image, "i"):
        plug_save_image.i = 0

    os.makedirs(output_dir, exist_ok=True)

    img = img_tensor.detach().cpu().squeeze(0)        # (3, 96, 96)
    if img.min() < 0:                                 # handle [-1, 1] normalized input
        img = (img + 1) / 2
    if img.max() <= 1.0:                              # float in [0, 1] -> [0, 255]
        img = img * 255
    img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()  # (96, 96, 3)

    path = os.path.join(output_dir, f"{plug_save_image.i}.jpeg")
    Image.fromarray(img).save(path, "JPEG")

    plug_save_image.i += 1
    return path










#x = torch.rand(1, 3, 96, 96)
#plug_save_image(x)   # saves output/0.jpeg
#plug_save_image(x)   # saves output/1.jpeg
