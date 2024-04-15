import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt 

# Load the saved model
model = tf.keras.models.load_model('D:/classification/lr_0.001_models/model_lr_0.001_epoch_7_20240412-153018.keras')

# Define the directory containing your test data
test_data_dir = 'C:/Users/Daniel/Downloads/test_ant_data/test_ant_data'

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
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Plot some performance metrics (example: ROC curve)
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
