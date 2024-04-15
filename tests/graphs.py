import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

model_dirs = [
    {'path': 'D:/classification/32', 'label': 'Batch Size = 32'},
    {'path': 'D:/classification/64', 'label': 'Batch Size = 64'},
]

# Initialize lists to store accuracies and epoch numbers for each folder
all_accuracies = [[] for _ in range(len(model_dirs))]
all_epochs = [[] for _ in range(len(model_dirs))]

# Iterate over the model directories
for dir_idx, model_info in enumerate(model_dirs):
    model_dir = model_info['path']
    # Initialize lists to store accuracies and epoch numbers for the current folder
    accuracies = []
    epochs = []

    # Iterate over the saved models in the current folder
    for model_file_idx, model_file in enumerate(sorted(os.listdir(model_dir))):
        if model_file.endswith('.keras'):
            # Load the model
            model = tf.keras.models.load_model(os.path.join(model_dir, model_file))

            # Extract epoch number from the model filename
            epoch = model_file_idx + 1

            # Initialize lists to store predictions and true labels
            predictions = []
            true_labels = []

            # Iterate over the folders containing test images
            test_data_dir = 'C:/Users/Daniel/Downloads/test_ant_data/test_ant_data'

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
                        # Normalize pixel values to [0, 1]
                        img_array = img_array / 255.0
                        # Make predictions
                        prediction = model.predict(img_array)[0][0]
                        predictions.append(prediction)
                        true_labels.append(class_label)

            # Convert predictions to class labels
            predicted_labels = (np.array(predictions) > 0.5).astype(int)

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, predicted_labels)
            accuracies.append(accuracy)
            epochs.append(epoch)
            
            # Print progress
            print(f"Completed testing epoch {epoch} of folder {dir_idx + 1}")

            # Break loop if reached the desired number of tests
            if len(epochs) >= 15:
                break

    # Store accuracies and epochs for the current folder
    all_accuracies[dir_idx] = accuracies
    all_epochs[dir_idx] = epochs

# Plot accuracy vs. epoch number for each folder
plt.figure(figsize=(10, 6))
for idx, (accuracies, epochs) in enumerate(zip(all_accuracies, all_epochs)):
    plt.plot(epochs, accuracies, marker='o', label=model_dirs[idx]['label'])

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.grid(True)
plt.legend()
plt.show()
