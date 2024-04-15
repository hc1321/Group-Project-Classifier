import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Flatten, Dense
from datetime import datetime
 

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
 

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'd:/finalData/trainingdata',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary' 
)
 
 
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
 
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
 
])
 
 
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False
 

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
 

model.fit(
    train_dataset,
 
    epochs=10
)
 

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'model_{current_time}.keras')