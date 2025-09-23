"""
src/infer.py

Image preprocessing and inference helper for the NumPy MLP.
Functions:
 - preprocess_image_pil(img: PIL.Image) -> np.ndarray (1,784)
 - load_params(path) -> params dict
 - predict_from_image_array(x_flat, params) -> (pred:int, probs:np.ndarray)
 - predict_from_image_path(path, params_path) -> (pred, probs)

This file is framework-agnostic and contains no web server logic.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from PIL import Image, ImageOps
from src.model.index import forward


CHECKPOINT_PATH = "saved_params.npz"


def preprocess_image_pil(img: Image.Image, debug_save: str | None = None) -> np.ndarray:
    """
    Robust preprocessing (Pillow + NumPy only) converting an input PIL image
    into a MNIST-like flattened array shape (1, 784), values in [0,1].

    debug_save: optional path to save the final 28x28 image for debugging.
    """
    # 1) fix EXIF orientation, convert to grayscale
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("L")

    # 2) convert to numpy array
    arr = np.array(img).astype(np.uint8)

    # 3) estimate background using border sampling (robust for screenshots)
    top = arr[0, :]
    bottom = arr[-1, :]
    left = arr[:, 0]
    right = arr[:, -1]
    border_vals = np.concatenate([top, bottom, left, right])
    bg_median = np.median(border_vals)

    # If borders are bright, invert so the digit/foreground becomes bright
    if bg_median > 127:
        arr = 255 - arr
        img = Image.fromarray(arr)

    # 4) threshold to get mask (robust but simple)
    maxv = arr.max() if arr.size else 0
    thr = max(maxv * 0.15, np.mean(arr) * 0.5)
    mask = arr > thr
    if mask.sum() == 0:
        thr = np.mean(arr) * 0.3
        mask = arr > thr

    # 5) get bbox from mask
    ys, xs = np.where(mask)
    if len(xs) == 0:
        left, top, right, bottom = 0, 0, arr.shape[1], arr.shape[0]
    else:
        left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

    # small pad to avoid clipped strokes
    pad = 2
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(arr.shape[1], right + pad)
    bottom = min(arr.shape[0], bottom + pad)

    # 6) crop and make square canvas, center the crop
    crop = arr[top:bottom, left:right]
    if crop.size == 0:
        crop = arr
    h, w = crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = crop

    # 7) resize to 20x20 (high-quality) and paste into 28x28 with 4-pixel margin
    small = Image.fromarray(canvas).resize((20, 20), Image.LANCZOS) # type: ignore
    final = Image.new('L', (28, 28), color=0)
    final.paste(small, (4, 4))

    # 8) center by intensity (approx center-of-mass)
    final_arr = np.array(final).astype(np.float32)
    total = final_arr.sum()
    if total > 0:
        rows = final_arr.sum(axis=1)
        cols = final_arr.sum(axis=0)
        cy = (np.arange(28) * rows).sum() / total
        cx = (np.arange(28) * cols).sum() / total
        shiftx = int(round(14 - cx))
        shifty = int(round(14 - cy))
        final_arr = np.roll(final_arr, shift=(shifty, shiftx), axis=(0, 1))

    # 9) normalize to [0,1] and flatten
    final_arr = np.clip(final_arr, 0, 255).astype(np.float32) / 255.0

    if debug_save:
        Image.fromarray((final_arr * 255).astype(np.uint8)).save(debug_save)

    return final_arr.reshape(1, -1)


def load_params(path: str = CHECKPOINT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}. Run training first to create saved_params.npz")
    data = np.load(path)
    params = {'W1': data['W1'], 'b1': data['b1'], 'W2': data['W2'], 'b2': data['b2']}
    return params


def predict_from_image_array(x_flat: np.ndarray, params: dict):
    probs, _ = forward(x_flat, params)
    pred = int(np.argmax(probs, axis=1)[0])
    return pred, probs[0]


def predict_from_image_path(image_path: str, checkpoint_path: str = CHECKPOINT_PATH):
    from PIL import Image
    params = load_params(checkpoint_path)
    img = Image.open(image_path)
    x = preprocess_image_pil(img)
    return predict_from_image_array(x, params)


# CLI helper when running this module directly
if __name__ == '__main__':
    import argparse
    # args = parser.parse_args()
    args = {}
    args["img"] = "image.webp"
    img = Image.open(args["img"])
    pred, probs = predict_from_image_path(args["img"], "saved_params.npz")
    print('Predicted:', pred)
    print('Top probs:', np.argsort(probs)[-3:][::-1], probs[np.argsort(probs)[-3:][::-1]])