import tensorflow as tf
from keras.applications import ResNet50 #change for ResNet50 or 101
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import Callback
from datetime import datetime
from keras.optimizers import Adam

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 34


train_dataset = tf.keras.utils.image_dataset_from_directory(
    'train',
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


model = Sequential([
    data_augmentation,
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

adam_opt = Adam(learning_rate = 0.01 )
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


test_data_directory = 'test'
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_directory,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'  
)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(self.test_data, verbose=0)
        print(f'\nEpoch {epoch+1}: Test loss: {test_loss}, Test accuracy: {test_accuracy}')


test_callback = TestCallback(test_data=test_dataset)

model.fit(
    train_dataset,
    epochs=1,
    callbacks=[test_callback]  
)

# Save the model
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'model_{current_time}.keras')


