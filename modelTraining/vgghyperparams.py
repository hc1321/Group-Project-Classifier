import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from datetime import datetime

IMAGE_SIZE = (128, 128)
BASE_BATCH_SIZE = 32
BASE_LR = 0.001
BASE_EPOCHS = 15

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'd:/finalData/trainingdata',
    target_size=IMAGE_SIZE,
    batch_size=BASE_BATCH_SIZE,
    class_mode='binary'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False

for lr in [0.01]:
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

   
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BASE_BATCH_SIZE,
        epochs=BASE_EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'model_lr_{lr}_epoch_{{epoch}}_{current_time}.keras',
                save_best_only=False,  
                save_weights_only=False,  
                monitor='loss',  
                mode='min',
                verbose=1
            )
        ]
    )


for batch_size in [16, 32, 64]:
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=BASE_EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'model_batch_{batch_size}_epoch_{{epoch}}_{current_time}.keras',
                save_best_only=False,  
                save_weights_only=False,  
                monitor='loss',
                mode='min',
                verbose=1
            )
        ]
    )
