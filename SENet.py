import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json

# Enable mixed precision - updated import
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy) # Use tf.keras to set policy

# Load datasets
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
(x_train_fashion_mnist, y_train_fashion_mnist), (x_test_fashion_mnist, y_test_fashion_mnist) = fashion_mnist.load_data()
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

# Normalize pixel values to between 0 and 1
x_train_mnist = x_train_mnist.astype('float32') / 255.0
x_test_mnist = x_test_mnist.astype('float32') / 255.0
x_train_fashion_mnist = x_train_fashion_mnist.astype('float32') / 255.0
x_test_fashion_mnist = x_test_fashion_mnist.astype('float32') / 255.0
x_train_cifar10 = x_train_cifar10.astype('float32') / 255.0
x_test_cifar10 = x_test_cifar10.astype('float32') / 255.0

# Adjust input shape for different datasets
x_train_mnist = np.expand_dims(x_train_mnist, axis=-1)  # Add channel dimension for MNIST
x_test_mnist = np.expand_dims(x_test_mnist, axis=-1)
x_train_fashion_mnist = np.expand_dims(x_train_fashion_mnist, axis=-1)  # Add channel dimension for Fashion MNIST
x_test_fashion_mnist = np.expand_dims(x_test_fashion_mnist, axis=-1)
# CIFAR-10 already has 3 channels (RGB)
input_shape_cifar10 = x_train_cifar10.shape[1:]

# Function to create SENet model
def build_senet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Squeeze and Excitation Block
    squeeze = layers.GlobalAveragePooling2D()(x)
    excitation = layers.Dense(64, activation='relu')(squeeze)
    excitation = layers.Dense(128, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, 128))(excitation)
    x = layers.multiply([x, excitation])
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate SENet model
def train_senet(dataset_name, x_train, y_train, x_test, y_test):
    if dataset_name == 'CIFAR-10':
        input_shape = input_shape_cifar10
    else:
        input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    model = build_senet(input_shape, num_classes)
    history = model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=128, validation_data=(x_test, to_categorical(y_test)), verbose=1)
    
    # Evaluate the model
    scores = model.evaluate(x_test, to_categorical(y_test), verbose=0)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{dataset_name} - Test Loss: {scores[0]}, Test Accuracy: {scores[1]}")
    print(f"{dataset_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{dataset_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'{dataset_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'test_loss': scores[0],
        'test_accuracy': scores[1],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Train and evaluate SENet model on each dataset
mnist_scores = train_senet('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)
fashion_mnist_scores = train_senet('Fashion MNIST', x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist)
cifar10_scores = train_senet('CIFAR-10', x_train_cifar10, y_train_cifar10, x_test_cifar10, y_test_cifar10)

# Save the scores to a JSON file
all_scores = {
    'MNIST': mnist_scores,
    'Fashion MNIST': fashion_mnist_scores,
    'CIFAR-10': cifar10_scores
}

with open('senet_scores.json', 'w') as f:
    json.dump(all_scores, f, indent=4)

# Save the scores to a text file
with open('senet_scores.txt', 'w') as f:
    for dataset, scores in all_scores.items():
        f.write(f"{dataset} Scores:\n")
        for metric, value in scores.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")