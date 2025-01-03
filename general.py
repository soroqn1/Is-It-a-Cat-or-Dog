import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from datasets import load_dataset
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка датасета
ds = load_dataset("nedith22/cats_and_dogs")

# change the dataset to a format suitable for Keras
def preprocess_data(dataset):
    images = []
    labels = []
    for item in dataset:
        image = item['image'].resize((150, 150))
        image = np.array(image)
        # random flip left-right
        image = tf.image.random_flip_left_right(image)
        images.append(image)
        labels.append(item['labels'])  # change this to 'label' if you are using a single label
    return tf.data.Dataset.from_tensor_slices((images, labels))

train_dataset = preprocess_data(ds['train'])
validation_dataset = preprocess_data(ds['test'])

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Замораживаем базовые слои

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_dataset.batch(20),
    epochs=30,  # epochs 
    validation_data=validation_dataset.batch(20)
)

model.save('cats_and_dogs_classifier.h5')
