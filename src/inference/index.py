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

import os
import numpy as np
from PIL import Image, ImageOps
from src.model.index import forward


CHECKPOINT_PATH = "saved_params.npz"


def preprocess_image_pil(img: Image.Image) -> np.ndarray:
    """Convert PIL image to flattened MNIST-like 28x28 numpy array (shape (1,784)).

    Steps:
      - convert to grayscale
      - trim whitespace (bounding box) if possible
      - resize keeping aspect (pad to square)
      - invert if necessary to match MNIST (white stroke on black)
      - normalize to [0,1]
      - flatten to shape (1, 784)
    """
    img = img.convert("L")
    arr = np.array(img)

    # Decide if inversion is needed: MNIST digits are bright on dark background
    if np.median(arr) > 127:
        arr = 255 - arr
        img = Image.fromarray(arr)

    # Crop to bounding box of content
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)

    # Create square canvas containing the image centered
    max_side = max(img.size)
    square = Image.new('L', (max_side, max_side), color=0)
    square.paste(img, ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2))

    # Resize to 20x20 and paste into 28x28 to mimic MNIST centering
    small = square.resize((20, 20)) # type: ignore
    final = Image.new('L', (28, 28), color=0)
    final.paste(small, (4, 4))

    arr = np.array(final).astype(np.float32) / 255.0
    flat = arr.reshape(1, -1)
    return flat


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to image file')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    args = parser.parse_args()

    pred, probs = predict_from_image_path(args.img, args.checkpoint)
    print('Predicted:', pred)
    print('Top probs:', np.argsort(probs)[-3:][::-1], probs[np.argsort(probs)[-3:][::-1]])