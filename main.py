import tensorflow as tf
import tensorflow.python.keras as keras
import pandas as pd
import numpy as np
import scipy
import os
import shutil
import matplotlib.pyplot as plt

from keras import layers

from keras.utils.np_utils import to_categorical

from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator


def add_to_class_directories(label, image_path, class_directory_path):
    new_image_path = class_directory_path + "/" + str(label)
    Path(new_image_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, new_image_path + "/" + image_path.split("train/", 1)[1])


def create_class_directories(csv_path, class_directory_path):
    df = pd.read_csv(csv_path)
    print(df)
    for row in df.iterrows():
        label = row[1][0]
        path = row[1][4]
        add_to_class_directories(label, path, class_directory_path)


data_dir = "train_test_data/train"

create_class_directories("train.csv", data_dir)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(322, 322),
    batch_size=32,
    class_mode='categorical')

model = keras.Sequential([
    layers.Resizing(83, 83),
    layers.Conv2D(64, 7, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 5, padding='same', activation='relu'),
    layers.Conv2D(128, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(256, 5, padding='same', activation='relu'),
    layers.Conv2D(256, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit_generator(
    train_generator,
    epochs=10,
    steps_per_epoch=43
)