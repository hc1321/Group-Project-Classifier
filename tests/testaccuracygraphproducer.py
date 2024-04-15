import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

# Define the directory containing your test data
test_data_dir = 'C:/Users/Daniel/Downloads/test_ant_data/test_ant_data'

# Initialize lists to store accuracies and epoch lengths
accuracies = []
epochs = []

# Iterate over the saved models
for idx, model_file in enumerate(sorted(os.listdir('D:/classification/lr_0.0001_models'))):
    if model_file.endswith('.keras'):
        # Load the model
        model = tf.keras.models.load_model(os.path.join('D:/classification/lr_0.0001_models', model_file))

        # Initialize lists to store images and true labels
        images = []
        true_labels = []

        # Iterate over the folders containing images
        for class_folder in os.listdir(test_data_dir):
            class_path = os.path.join(test_data_dir, class_folder)
            if os.path.isdir(class_path):
                # Assuming class folders are named '0' and '1'
                class_label = int(class_folder)
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    img = image.load_img(img_path, target_size=(128, 128))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    images.append(img_array)
                    true_labels.append(class_label)

        # Stack the images into a single numpy array
        images = np.vstack(images)

        # Normalize pixel values to [0, 1]
        images = images / 255.0

        # Make predictions on the test data
        predictions = model.predict(images)

        # Convert probabilities to class labels
        predicted_labels = (predictions > 0.5).astype(int).flatten()

        # Calculate accuracy using scikit-learn's accuracy_score function
        accuracy = accuracy_score(true_labels, predicted_labels)
        accuracies.append(accuracy)
        epochs.append(idx + 1)  # Count starts from 1

# Plot accuracy vs. epoch number
plt.plot(epochs, accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.grid(True)
plt.show()
