import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json

# Enable mixed precision
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Load datasets
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
(x_train_fashion_mnist, y_train_fashion_mnist), (x_test_fashion_mnist, y_test_fashion_mnist) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Resize MNIST and Fashion MNIST images to 71x71 using tf.image.resize
x_train_mnist = tf.image.resize(np.expand_dims(x_train_mnist, axis=-1), [71, 71]).numpy()
x_test_mnist = tf.image.resize(np.expand_dims(x_test_mnist, axis=-1), [71, 71]).numpy()
x_train_fashion_mnist = tf.image.resize(np.expand_dims(x_train_fashion_mnist, axis=-1), [71, 71]).numpy()
x_test_fashion_mnist = tf.image.resize(np.expand_dims(x_test_fashion_mnist, axis=-1), [71, 71]).numpy()
x_train_cifar = tf.image.resize(x_train, [71, 71]).numpy()
x_test_cifar = tf.image.resize(x_test, [71, 71]).numpy()


# Preprocess datasets
x_train_mnist = x_train_mnist / 255.0
x_test_mnist = x_test_mnist / 255.0
x_train_fashion_mnist = x_train_fashion_mnist / 255.0
x_test_fashion_mnist = x_test_fashion_mnist / 255.0
x_train = x_train_cifar.astype('float32') / 255.0
x_test = x_test_cifar.astype('float32') / 255.0


x_train_mnist = x_train_mnist[:int(0.99 * len(x_train_mnist))]
y_train_mnist = y_train_mnist[:int(0.99 * len(y_train_mnist))]
x_train_fashion_mnist = x_train_fashion_mnist[:int(0.99 * len(x_train_fashion_mnist))]
y_train_fashion_mnist = y_train_fashion_mnist[:int(0.99 * len(y_train_fashion_mnist))]

# Function to create Xception model
def build_xception(input_shape, num_classes):
    base_model = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # Ensuring the output is float32
    model = models.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate Xception model
def train_xception(dataset_name, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    with strategy.scope():
        model = build_xception(input_shape, num_classes)
    
    history = model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=64 * strategy.num_replicas_in_sync, validation_data=(x_test, to_categorical(y_test)), verbose=1)
    
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

# Train and evaluate Xception model on MNIST and Fashion MNIST datasets
mnist_scores = train_xception('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)
fashion_mnist_scores = train_xception('Fashion MNIST', x_train_fashion_mnist, y_train_fashion_mnist, x_test_fashion_mnist, y_test_fashion_mnist)
cifar10_scores = train_xception(x_train_resized, y_train, x_test_resized, y_test)

# Save the scores to a JSON file
all_scores = {
    'MNIST': mnist_scores,
    'Fashion MNIST': fashion_mnist_scores,
    'Cifar-10': cifar10_scores
}

# Save the scores to a text file
with open('xception_scores.txt', 'w') as f:
    for dataset, scores in all_scores.items():
        f.write(f"{dataset} Scores:\n")
        for metric, value in scores.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")