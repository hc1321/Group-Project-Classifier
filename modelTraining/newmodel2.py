import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

def load_images_and_labels(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (150, 150)) 
            data.append(img)
            labels.append(label)

    return data, labels


folder_with_leaves = "D:\\finalData\\leaves512"
folder_without_leaves = "D:\\finalData\\noleaves512"


data_with_leaves, labels_with_leaves = load_images_and_labels(folder_with_leaves, 1)
data_without_leaves, labels_without_leaves = load_images_and_labels(folder_without_leaves, 0)

# Combine data and labels
X = np.array(data_with_leaves + data_without_leaves)
y = np.array(labels_with_leaves + labels_without_leaves)
  
# Normalize pixel values
X = X / 255.0

# Define data generators
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_datagen.fit(X)

test_datagen = ImageDataGenerator()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = models.Sequential()

# Convolutional layers with maxpooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Dense layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    epochs=20,
    validation_data=test_datagen.flow(X_test, y_test),
    validation_steps=len(X_test) / 32,
    callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
