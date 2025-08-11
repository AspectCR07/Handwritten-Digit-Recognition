"""
model.py
Defines the CNN model for MNIST and helper functions to build / compile.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(28,28,1), n_classes=10, dropout_rate=0.25):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def compile_model(model, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
