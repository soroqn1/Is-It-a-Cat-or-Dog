import tensorflow as tf
from tensorflow.keras.models import load_model
from datasets import load_dataset
import numpy as np
from PIL import Image
import os

model = load_model('cats_and_dogs_classifier.h5')

ds = load_dataset("nedith22/cats_and_dogs")

# change the dataset to a format suitable for Keras
def preprocess_data(dataset):
    images = []
    labels = []
    for item in dataset:
        image = item['image'].resize((150, 150))
        image = np.array(image)
        images.append(image)
        labels.append(item['labels'])
    return tf.data.Dataset.from_tensor_slices((images, labels))

test_dataset = preprocess_data(ds['test'])

# assesment of the model
test_loss, test_accuracy = model.evaluate(test_dataset.batch(20))
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# your image prediction
def predict_on_custom_image(image_path):
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        return
    try:
        image = Image.open(image_path)
        print(f"Original image size: {image.size}")  # (width, height)
        image = image.resize((150, 150))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # adding batch dimension
        prediction = model.predict(image)
        print(f'Predicted: {prediction[0][0]}')
        if prediction[0][0] < 0.5:
            print("The neural network suggests that it's a CAT!")
        else:
            print("The neural network suggests that it's a DOG!")
    except Exception as e:
        print(f"Error loading image: {e}")

custom_image_path = 'PATH_TO_YOUR_IMAGE' # change this to the path of your image
predict_on_custom_image(custom_image_path)
