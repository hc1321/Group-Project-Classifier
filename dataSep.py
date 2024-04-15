import os
import shutil
from sklearn.model_selection import train_test_split

#note only run this once, if re-run there will be duplicates

def split_data(source_folder, dest_folder, test_split=0.2):

    val_dir = os.path.join(dest_folder, 'val')
    test_dir = os.path.join(dest_folder, 'test')
    
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_folder in ['noleaves', 'leaves']:

        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

        class_dir = os.path.join(source_folder, class_folder)

        images = os.listdir(class_dir)

        val_images, test_images = train_test_split(images, test_size=test_split, random_state=42)
        
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, class_folder, img))        
        
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, class_folder, img))

split_data('realData', '.', test_split=0.2)
