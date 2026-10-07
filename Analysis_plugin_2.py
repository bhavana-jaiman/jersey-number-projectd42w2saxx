import os
import torch
from PIL import Image

def plug_save_image_batch(img_tensor, output_dir="output", batch_idx=None):
    """Save a (B, 3, H, W) batch as <batch_idx>_<i>.jpeg, where i is a running image index."""
    if not hasattr(plug_save_image_batch, "i"):
        plug_save_image_batch.i = 0
    if not hasattr(plug_save_image_batch, "batch_idx"):
        plug_save_image_batch.batch_idx = 0

    # use the batch index passed in, otherwise the internal counter
    b = plug_save_image_batch.batch_idx if batch_idx is None else batch_idx

    os.makedirs(output_dir, exist_ok=True)

    if img_tensor.dim() == 3:                         # allow a single (3, H, W) image too
        img_tensor = img_tensor.unsqueeze(0)

    batch = img_tensor.detach().cpu()
    paths = []

    for img in batch:                                 # img: (3, H, W)
        if img.min() < 0:                             # handle [-1, 1] normalized input
            img = (img + 1) / 2
        if img.max() <= 1.0:                          # float in [0, 1] -> [0, 255]
            img = img * 255
        img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()   # (H, W, 3)

        path = os.path.join(output_dir, f"{b}_{plug_save_image_batch.i}.jpeg")
        Image.fromarray(img).save(path, "JPEG")
        paths.append(path)

        plug_save_image_batch.i += 1

    plug_save_image_batch.batch_idx += 1
    return paths



#Usage with automatic batch counting:
#plug_save_image_batch(batch, "Testing3")   # 0_0.jpeg ... 0_7.jpeg
#plug_save_image_batch(batch, "Testing3")   # 1_8.jpeg ... 1_15.jpeg
#pass the batch index from your DataLoader loop,
'''for batch_idx, (images, labels) in enumerate(loader):
    plug_save_image_batch(images, "Testing3", batch_idx=batch_idx)'''
