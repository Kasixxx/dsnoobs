# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Use a breakpoint in the code line below to debug your script.
# Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import tensorflow as tf
import tensorflow.python.keras as keras
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

from keras import layers

from pathlib import Path

data_dir = "/Users/14kas/PycharmProjects/pythonProject1/train_test_data/train"


def split_data_set(dataset):
    for images_df, labels_df in dataset.take(1):
        numpy_images = images_df.numpy()
        numpy_labels = labels_df.numpy()
        return numpy_images, numpy_labels


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


create_class_directories("train.csv", "train_test_data/train")

image_df_train, image_df_validation = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=2137,
    validation_split=0.2,
    subset="both",
    image_size=(332, 332)
)

model = keras.Sequential([
    layers.Flatten(),
    layers.Normalization(),
    layers.Dense(256, activation='relu', input_shape=(332 * 332,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

images, labels = split_data_set(image_df_train)

model.fit(
    images,
    labels,
    epochs=5,
    batch_size=30,
)
