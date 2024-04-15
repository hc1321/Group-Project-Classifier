import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import h5py

MODEL_PATH = 'model_20240415-181135.keras'   #fill with model to test
TEST_DATA_DIRECTORY = 'test' 
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 34  


if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
print(model.summary())


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIRECTORY,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)
for images, labels in test_dataset.take(1):
    print('Images batch shape:', images.shape)
    print('Labels batch shape:', labels.shape)
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
