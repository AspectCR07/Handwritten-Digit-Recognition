"""
predict.py
Predict digits from image files (PNG / JPG). Expects images of handwritten digits,
but will resize/normalize to 28x28 grayscale.

Example:
  python predict.py --model models/mnist_cnn_best.h5 --files samples/digit1.png samples/digit2.png
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from utils import load_and_preprocess_image
import os

def main(args):
    model = load_model(args.model)
    files = args.files
    for f in files:
        if not os.path.exists(f):
            print("File not found:", f); continue
        img = load_and_preprocess_image(f)  # (28,28,1)
        x = np.expand_dims(img, 0)  # batch
        pred = model.predict(x)
        label = int(np.argmax(pred, axis=1)[0])
        prob = float(np.max(pred))
        print(f"{os.path.basename(f)} -> Predicted: {label}  confidence: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model (.h5)')
    parser.add_argument('files', nargs='+', help='Image file(s) to predict')
    args = parser.parse_args()
    main(args)
