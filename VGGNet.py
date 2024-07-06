import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# Function to build VGGNet model
def build_vggnet(input_shape, num_classes):
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 4
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 5
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Function to preprocess and load MNIST data
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    
    return (x_train, y_train, x_test, y_test)

# Function to preprocess and load Fashion MNIST data
def load_and_preprocess_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
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

# Function to train VGGNet on a dataset
def train_vggnet(dataset_name, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    model = build_vggnet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Training VGGNet on {dataset_name} dataset...")
    history = model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=128, validation_data=(x_test, to_categorical(y_test)), verbose=1)
    
    print(f"Evaluating VGGNet on {dataset_name} dataset...")
    test_loss, test_acc = model.evaluate(x_test, to_categorical(y_test), verbose=0)
    
    y_pred = model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test, y_pred_class, average='weighted')
    recall = recall_score(y_test, y_pred_class, average='weighted')
    f1 = f1_score(y_test, y_pred_class, average='weighted')
    
    metrics_text = f"Dataset: {dataset_name}\n" \
                   f"Test accuracy: {test_acc:.4f}\n" \
                   f"Precision: {precision:.4f}\n" \
                   f"Recall: {recall:.4f}\n" \
                   f"F1-score: {f1:.4f}\n"
    
    os.makedirs('metrics', exist_ok=True)
    with open(f"metrics/{dataset_name}_vggnet_metrics.txt", 'w') as f:
        f.write(metrics_text)
    
    print(metrics_text)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{dataset_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{dataset_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{dataset_name}_vggnet_training_history.png")
    plt.show()

# Load and preprocess datasets
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_and_preprocess_mnist()
x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist = load_and_preprocess_fashion_mnist()
x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_and_preprocess_cifar10()

# Train and evaluate VGGNet on MNIST
train_vggnet('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

# Train and evaluate VGGNet on Fashion MNIST
train_vggnet('Fashion_MNIST', x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist)

# Train and evaluate VGGNet on CIFAR-10
train_vggnet('CIFAR-10', x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)