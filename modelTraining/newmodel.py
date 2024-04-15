import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
import tensorflow as tf


def load_images_and_labels(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (128, 128)) 
            data.append(img)
            labels.append(label)

    return data, labels

def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch > 10:
        return initial_lr * 0.1
    elif epoch > 5:
        return initial_lr * 0.5
    else:
        return initial_lr


folder_with_leaves = r""
folder_without_leaves = r""


data_with_leaves, labels_with_leaves = load_images_and_labels(folder_with_leaves, 1)
data_without_leaves, labels_without_leaves = load_images_and_labels(folder_without_leaves, 0)


X = np.array(data_with_leaves + data_without_leaves)
y = np.array(labels_with_leaves + labels_without_leaves)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout with 50% probability
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[LearningRateScheduler(lr_schedule, verbose=1)])


test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
