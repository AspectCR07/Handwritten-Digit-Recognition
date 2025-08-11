"""
utils.py
Utilities for loading and preprocessing image files for prediction.
"""

from PIL import Image
import numpy as np

def load_and_preprocess_image(path, target_size=(28,28)):
    """
    Loads an image, converts to grayscale, resizes to target_size, inverts if needed,
    and scales to [0,1]. Returns shape (28,28,1).
    """
    img = Image.open(path).convert('L')  # grayscale
    img = img.resize(target_size, Image.ANTIALIAS)
    arr = np.array(img).astype('float32')
    # many handwritten samples have white background & black ink (MNIST is white-on-black? actually MNIST: black background, white digit)
    # normalize so that digit appears as bright values similar to MNIST (white on black)
    # If background is light (mean>127), invert
    if arr.mean() > 127:
        arr = 255 - arr
    arr /= 255.0
    arr = np.expand_dims(arr, -1)
    return arr
