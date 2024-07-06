import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import json

# Function to build a simple ResNet model
def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual block
    def res_block(x, filters):
        y = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        y = layers.Conv2D(filters, (3, 3), padding='same', activation=None)(y)
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, (1, 1), padding='same', activation=None)(x)
        return layers.add([x, y])
    
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = res_block(x, 256)
    x = res_block(x, 512)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x)
    return model

# Function to preprocess and load MNIST data
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=0.75, random_state=42)
    
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    
    return (x_train, y_train, x_test, y_test)

# Function to preprocess and load Fashion MNIST data
def load_and_preprocess_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=0.75, random_state=42)
    
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    
    return (x_train, y_train, x_test, y_test)

# Function to preprocess and load CIFAR-10 data
def load_and_preprocess_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train, x_test, y_test)

# Function to train and evaluate the ResNet model
def train_resnet(dataset_name, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    model = build_resnet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=128, validation_data=(x_test, to_categorical(y_test)), verbose=1)
    
    scores = model.evaluate(x_test, to_categorical(y_test), verbose=0)

    # Evaluate the model
    scores = model.evaluate(x_test, to_categorical(y_test), verbose=0)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{dataset_name} ResNet Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{dataset_name} ResNet Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plots as PNG
    plt.savefig(f"{dataset_name.lower()}_resnet_training_history.png")
    plt.show()
    
    return {
        'test_accuracy': scores[1],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Load and preprocess data for each dataset
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_and_preprocess_mnist()
x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist = load_and_preprocess_fashion_mnist()
x_train_cifar10, y_train_cifar10, x_test_cifar10, y_test_cifar10 = load_and_preprocess_cifar10()

# Train and evaluate ResNet model on each dataset
mnist_scores = train_resnet('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)
fashion_mnist_scores = train_resnet('Fashion MNIST', x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist)
cifar10_scores = train_resnet('CIFAR-10', x_train_cifar10, y_train_cifar10, x_test_cifar10, y_test_cifar10)

# Save the scores to a text file
all_scores = {
    'MNIST': mnist_scores,
    'Fashion MNIST': fashion_mnist_scores,
    'CIFAR-10': cifar10_scores
}

with open('resnet_scores.json', 'w') as f:
    json.dump(all_scores, f, indent=4)