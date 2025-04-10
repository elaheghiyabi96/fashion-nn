# fashion-nn
A simple neural network for classifying Fashion MNIST images using TensorFlow.
🧠 Fashion MNIST Classification with TensorFlow
This project demonstrates how to build and train a simple Neural Network using TensorFlow and Keras to classify images from the Fashion MNIST dataset.

📊 Dataset
The Fashion MNIST dataset contains 70,000 grayscale images (28x28 pixels) of 10 different types of clothing, such as t-shirts, trousers, shoes, bags, etc.

60,000 images for training

10,000 images for testing

Labels (classes):
['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

🏗️ Model Architecture
The neural network is a simple feedforward (dense) model using Keras Sequential() API:

Input layer: 784 neurons (flattened 28x28 image)

Hidden layer 1: 128 neurons with ReLU activation

Hidden layer 2: 64 neurons with ReLU activation

Output layer: 10 neurons with Softmax activation

⚙️ Compilation & Training
Loss Function: Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy

Epochs: 20

Batch size: 32

Validation: Performed on test data

📈 Visualization
After training, the loss (both training and validation) is plotted over the epochs to visualize how the model learned.
🏁 Result
Achieved a test accuracy of around 88.47% using this simple neural network.

#DeepLearning 
#TensorFlow 
#FashionMNIST 
#ImageClassification 
#NeuralNetworks 
#MachineLearning 
#Keras
# Importing TensorFlow library (used for building and training neural networks)
import tensorflow as tf

# Importing necessary classes for building a Sequential model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Importing the Fashion MNIST dataset (images of clothing items)
from tensorflow.keras.datasets import fashion_mnist

# Importing utility to convert labels into one-hot encoded format
from tensorflow.keras.utils import to_categorical

# Importing categorical cross-entropy loss function for multi-class classification
from tensorflow.keras.losses import CategoricalCrossentropy

# Importing matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Loading Fashion MNIST dataset and splitting into training and testing sets
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# Normalizing image pixel values to range [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshaping images from 28x28 to 784 (flattening each image)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Converting integer labels to one-hot encoded vectors (10 classes)
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Defining a Sequential neural network model with 3 layers:
# - First hidden layer with 128 neurons and ReLU activation
# - Second hidden layer with 64 neurons and ReLU activation
# - Output layer with 10 neurons (one per class) and softmax activation
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(784,)),
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Compiling the model:
# - Optimizer: Adam (adaptive learning rate)
# - Loss function: Categorical Crossentropy (for multi-class classification)
# - Metrics: Accuracy (to evaluate performance)
model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Training the model for 20 epochs with batch size of 32
# Also evaluating on validation (test) data during training
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

# Evaluating the trained model on test data to get final loss and accuracy
loss, accuracy = model.evaluate(X_test, Y_test)

# Printing final accuracy as a percentage
print(f" Accuracy : {accuracy * 100:.2f}%")

# Plotting the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Importing matplotlib to display images
import matplotlib.pyplot as plt

# Importing numpy to perform numerical operations (like argmax)
import numpy as np

# Defining the class names corresponding to the labels in Fashion MNIST
# These are human-readable names for each category (0 to 9)
class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'coat', 'sandal', 'Shirt', 'sneaker', 'bag', 'ankle boot']

# Selecting the index of the image you want to test (e.g., image number 200)
index = 200

# Making a prediction for that specific image
# We reshape it to (1, 784) because model expects input shape as a batch
prediction = model.predict(X_test[index].reshape(1, 28 * 28))

# Displaying the selected image in 2D (28x28)
plt.imshow(X_test[index].reshape(28, 28))
plt.show()

# Printing the actual label name for the selected image
# np.argmax(Y_test[index]) gives the class index; we use it to look up the class name
print("Actual label:" , class_name[np.argmax(Y_test[index])])

