import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

def load_images_and_labels(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (224, 224))
            data.append(img)
            labels.append(label)

    return data, labels


folder_with_leaves = r""
folder_without_leaves = r""


data_with_leaves, labels_with_leaves = load_images_and_labels(folder_with_leaves, 1)
data_without_leaves, labels_without_leaves = load_images_and_labels(folder_without_leaves, 0)

data_with_leaves = np.array(data_with_leaves)
data_without_leaves = np.array(data_without_leaves)
labels_with_leaves = np.array(labels_with_leaves)
labels_without_leaves = np.array(labels_without_leaves)


data = np.concatenate([data_with_leaves, data_without_leaves], axis=0)
labels = np.concatenate([labels_with_leaves, labels_without_leaves], axis=0)


train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)


train_data = train_data / 255.0
val_data = val_data / 255.0


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 5:
        lr *= 0.1
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)


history = model.fit(
    datagen.flow(train_data, train_labels, batch_size=32),
    steps_per_epoch=len(train_data) // 32,
    epochs=20,
    validation_data=(val_data, val_labels),
    callbacks=[lr_scheduler]
)
