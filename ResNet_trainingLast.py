import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import Callback
from datetime import datetime
from keras.optimizers import Adam

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 34


train_dataset = tf.keras.utils.image_dataset_from_directory(
    'finalData/train',
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

#base_model.summary()

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    if 'conv5_block' in layer.name:
        layer.trainable = True
    else:
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
               metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


test_data_directory = 'finalData/test'
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
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.accuracy = tf.keras.metrics.BinaryAccuracy()

    def on_epoch_end(self, epoch, logs=None):
        self.precision.reset_state()
        self.recall.reset_state()
        self.accuracy.reset_state()
        for batch in self.test_data:
            x_test, y_test = batch
            y_pred = self.model.predict(x_test)
            self.precision.update_state(y_test, y_pred)
            self.recall.update_state(y_test, y_pred)
            self.accuracy.update_state(y_test, y_pred)
        test_precision = self.precision.result()
        test_recall = self.recall.result()
        test_accuracy = self.accuracy.result()
        print(f'\nEpoch {epoch+1}: Test accuracy: {test_accuracy:.4f}, Test precision: {test_precision:.4f}, Test recall: {test_recall:.4f}')


test_callback = TestCallback(test_data=test_dataset)

model.fit(
    train_dataset,
    epochs=20,
    callbacks=[test_callback]  
)

# Save the model
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'model_{current_time}.keras')


