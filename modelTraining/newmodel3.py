import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split


def load_images_and_labels(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize if needed
            data.append(img)
            labels.append(label)

    return data, labels

folder_with_leaves = "withLeaves00041"
folder_without_leaves = "noLeaves00041"

data_with_leaves, labels_with_leaves = load_images_and_labels(folder_with_leaves, 1)
data_without_leaves, labels_without_leaves = load_images_and_labels(folder_without_leaves, 0)

X = np.array(data_with_leaves + data_without_leaves)
y = np.array(labels_with_leaves + labels_without_leaves)


X = X / 255.0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_test, X_unseen, y_test, y_unseen = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

base_model.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_datagen.fit(X_train)

test_datagen = ImageDataGenerator()


history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    epochs=20,
    validation_data=test_datagen.flow(X_test, y_test),
    validation_steps=len(X_test) / 32)


test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)


unseen_loss, unseen_acc = model.evaluate(X_unseen, y_unseen)
print("Unseen Data Accuracy:", unseen_acc)
