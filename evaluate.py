"""
evaluate.py
Load a saved model and evaluate on the MNIST test set, printing accuracy and a confusion matrix.
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    return (x_test, y_test)

def main(args):
    x_test, y_test = load_data()
    model = load_model(args.model)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")

    preds = model.predict(x_test, batch_size=256)
    y_pred = preds.argmax(axis=1)

    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix - MNIST")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out)
        print("Confusion matrix saved to", args.out)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved .h5 model')
    parser.add_argument('--out', default=None, help='Path to save confusion matrix image')
    args = parser.parse_args()
    main(args)
