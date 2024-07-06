import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to build AlexNet model with adjusted pooling layers
def build_alexnet(input_shape, num_classes):
    model = models.Sequential()
    
    # Layer 1
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    
    # Layer 2
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    
    # Layer 3
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    
    # Layer 4
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    
    # Layer 5
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Function to preprocess and load MNIST data
def load_and_preprocess_mnist():
    from tensorflow.keras.datasets import mnist
    
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    
    # Normalize data
    x_train_mnist = np.expand_dims(x_train_mnist, axis=-1).astype('float32') / 255.0
    x_test_mnist = np.expand_dims(x_test_mnist, axis=-1).astype('float32') / 255.0
    
    # Resize images to 32x32 to fit AlexNet input size
    x_train_mnist = np.pad(x_train_mnist, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test_mnist = np.pad(x_test_mnist, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    
    return (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

# Function to preprocess and load Fashion MNIST data
def load_and_preprocess_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist
    
    (x_train_fashion_mnist, y_train_fashion_mnist), (x_test_fashion_mnist, y_test_fashion_mnist) = fashion_mnist.load_data()
    
    # Normalize data
    x_train_fashion_mnist = np.expand_dims(x_train_fashion_mnist, axis=-1).astype('float32') / 255.0
    x_test_fashion_mnist = np.expand_dims(x_test_fashion_mnist, axis=-1).astype('float32') / 255.0
    
    # Resize images to 32x32 to fit AlexNet input size
    x_train_fashion_mnist = np.pad(x_train_fashion_mnist, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test_fashion_mnist = np.pad(x_test_fashion_mnist, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    
    return (x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist)

# Function to preprocess and load CIFAR-10 data
def load_and_preprocess_cifar10():
    from tensorflow.keras.datasets import cifar10
    
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
    
    # Normalize data
    x_train_cifar = x_train_cifar.astype('float32') / 255.0
    x_test_cifar = x_test_cifar.astype('float32') / 255.0
    
    return (x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)

# Function to train AlexNet on a dataset
def train_alexnet(dataset_name, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    # Build AlexNet model
    model = build_alexnet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print(f"Training AlexNet on {dataset_name} dataset...")
    history = model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=128, validation_data=(x_test, to_categorical(y_test)), verbose=1)
    
    # Evaluate the model
    print(f"Evaluating AlexNet on {dataset_name} dataset...")
    test_loss, test_acc = model.evaluate(x_test, to_categorical(y_test), verbose=0)
    
    # Calculate additional metrics
    y_pred = model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test, y_pred_class, average='weighted')
    recall = recall_score(y_test, y_pred_class, average='weighted')
    f1 = f1_score(y_test, y_pred_class, average='weighted')
    
    # Print and save evaluation metrics to a text file
    metrics_text = f"Dataset: {dataset_name}\n" \
                   f"Test accuracy: {test_acc:.4f}\n" \
                   f"Precision: {precision:.4f}\n" \
                   f"Recall: {recall:.4f}\n" \
                   f"F1-score: {f1:.4f}\n"
    
    with open(f"{dataset_name}_alexnet_metrics.txt", 'w') as f:
        f.write(metrics_text)
    
    print(metrics_text)
    
    # Plot training history
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
    
    plt.savefig(f"{dataset_name}_alexnet_training_history.png")
    plt.show()

# Load and preprocess datasets
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_and_preprocess_mnist()
x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist = load_and_preprocess_fashion_mnist()
x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_and_preprocess_cifar10()

# Train and evaluate AlexNet on MNIST
train_alexnet('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)

# Train and evaluate AlexNet on Fashion MNIST
train_alexnet('Fashion_MNIST', x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist)

# Train and evaluate AlexNet on CIFAR-10
train_alexnet('CIFAR-10', x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)