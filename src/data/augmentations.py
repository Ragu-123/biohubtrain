import random
import torch

def apply_d4_flips(img: torch.Tensor, coords: torch.Tensor = None, shape: tuple = None):
    dims = []
    if random.random() > 0.5: dims.append(-1)
    if random.random() > 0.5: dims.append(-2)
    if random.random() > 0.5: dims.append(-3)
    if not dims: return img, coords
    flipped = img.flip(dims=dims)
    if coords is not None and shape is not None:
        c = coords.clone()
        for d in dims:
            ax = d + 3
            c[:, ax] = shape[ax] - 1 - c[:, ax]
        return flipped, c
    return flipped, coords

def apply_brightness_shift(img: torch.Tensor, max_shift: float = 0.10):
    shift = random.uniform(-max_shift, max_shift)
    return (img + shift).clamp(0.0, 1.0)
