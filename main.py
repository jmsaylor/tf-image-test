import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import pathlib

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential

if __name__ == '__main__':
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, untar='True')
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    print(list(train_ds))

    class_names = train_ds.class_names
    print(class_names)

    # CIFAR Data Set
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    #
    # train_images, test_images = train_images / 255.0, test_images / 255.0

    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                'dog', 'frog', 'horse', 'ship', 'truck']

    #Constructing the Layers of a model
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(10))

    # model.summary()