"""
train.py
Load MNIST, train CNN with data augmentation, save best model (by validation accuracy).
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import build_cnn, compile_model

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def main(args):
    (x_train, y_train), (x_test, y_test) = load_data()

    # split validation from train
    val_split = args.val_split
    vcount = int(len(x_train) * val_split)
    x_val = x_train[:vcount]
    y_val = y_train[:vcount]
    x_tr = x_train[vcount:]
    y_tr = y_train[vcount:]

    print(f"Train: {x_tr.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.08
    )
    datagen.fit(x_tr, augment=True)

    # build model
    model = build_cnn(input_shape=(28,28,1), n_classes=10)
    model = compile_model(model, lr=args.lr)
    model.summary()

    # callbacks
    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, "mnist_cnn_best.h5")
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

    # training
    batch_size = args.batch_size
    epochs = args.epochs
    steps_per_epoch = int(np.ceil(len(x_tr) / batch_size))

    history = model.fit(
        datagen.flow(x_tr, y_tr, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=2
    )

    # final evaluation on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST CNN")
    parser.add_argument('--out-dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of train used as validation')
    args = parser.parse_args()
    main(args)
