# Cats and Dogs Classifier 🐶🐱

A machine learning project for classifying images of cats and dogs using TensorFlow and Keras with transfer learning.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/12cd5ccd-c8f4-44de-8fac-8b6ec59d11d6" />

## Description

This project leverages a convolutional neural network (CNN) with the MobileNetV2 architecture to classify images of cats and dogs. It includes data preprocessing, model training with data augmentation, evaluation, and the ability to predict on custom images.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/soroqn1/is-it-a-cat-or-dog
    cd it-a-cat-or-dog
    ```

2. **Set up a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the `general.py` script:

```bash
python3 general.py
```

**What It Does:**

- **Loads the Dataset:** Utilizes the `datasets` library to load the "nedith22/cats_and_dogs" dataset.
- **Preprocesses the Data:** Resizes images to 150x150 pixels and applies random horizontal flipping for data augmentation.
- **Defines the Model:** Uses MobileNetV2 as the base model with frozen layers, adds global average pooling, dropout for regularization, and a dense layer for binary classification.
- **Compiles the Model:** Configured with the Adam optimizer and binary cross-entropy loss.
- **Trains the Model:** Runs for 30 epochs and validates on the test dataset.
- **Saves the Model:** Stores the trained model as `cats_and_dogs_classifier.h5`.

### Testing the Model

To evaluate the model and make predictions on custom images, run the `test_model.py` script:

```bash
python3 test_model.py
```

**What It Does:**

- **Loads the Trained Model:** Imports the saved `cats_and_dogs_classifier.h5` model.
- **Preprocesses the Test Data:** Resizes and formats test images for evaluation.
- **Evaluates the Model:** Prints out the test loss and accuracy.
- **Predicts on Custom Images:** Allows you to input a custom image path to determine if it's a cat or dog.

**Usage Example:**

Edit the `custom_image_path` variable in `test_model.py` to point to your image:

```python
custom_image_path = '/path/to/your/image.jpg'
```

Run the script to see the prediction:

```bash
python3 test_model.py
```

## Model Architecture

The model architecture is based on MobileNetV2, a lightweight and efficient deep learning model suitable for mobile and embedded vision applications. The architecture includes:

- **Base Model:** MobileNetV2 with pre-trained ImageNet weights (layers frozen).
- **Global Average Pooling:** Reduces the spatial dimensions.
- **Dropout Layer:** Prevents overfitting by randomly setting input units to 0.
- **Dense Layer:** Single neuron with sigmoid activation for binary classification.

## Results

After training, the model achieved the following performance on the test dataset:

- **Test Loss:** 0.6102
- **Test Accuracy:** 69.01%

**Notes:**

- The accuracy indicates moderate performance. Further improvements can be made by enhancing data augmentation, fine-tuning more layers of the base model, or increasing the dataset size.
- Implementing techniques like early stopping and learning rate scheduling can also help in achieving better results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
