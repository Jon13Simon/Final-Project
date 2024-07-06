import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    # Load datasets
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

    # Normalize the data to the range [0, 1]
    x_train_mnist, x_test_mnist = x_train_mnist / 255.0, x_test_mnist / 255.0
    x_train_fmnist, x_test_fmnist = x_train_fmnist / 255.0, x_test_fmnist / 255.0
    x_train_cifar, x_test_cifar = x_train_cifar / 255.0, x_test_cifar / 255.0

    # Reshape the data to add a channel dimension
    x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1)
    x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1)
    x_train_fmnist = x_train_fmnist.reshape(-1, 28, 28, 1)
    x_test_fmnist = x_test_fmnist.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    y_train_mnist = to_categorical(y_train_mnist)
    y_test_mnist = to_categorical(y_test_mnist)
    y_train_fmnist = to_categorical(y_train_fmnist)
    y_test_fmnist = to_categorical(y_test_fmnist)
    y_train_cifar = to_categorical(y_train_cifar)
    y_test_cifar = to_categorical(y_test_cifar)

    return (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist), \
           (x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist), \
           (x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)

# Load and preprocess the data
mnist_data, fmnist_data, cifar_data = load_and_preprocess_data()

# Extract preprocessed data
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = mnist_data
x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist = fmnist_data
x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = cifar_data

# Print shapes to verify
print("MNIST:", x_train_mnist.shape, y_train_mnist.shape, x_test_mnist.shape, y_test_mnist.shape)
print("Fashion MNIST:", x_train_fmnist.shape, y_train_fmnist.shape, x_test_fmnist.shape, y_test_fmnist.shape)
print("CIFAR-10:", x_train_cifar.shape, y_train_cifar.shape, x_test_cifar.shape, y_test_cifar.shape)