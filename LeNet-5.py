import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Function to define and train LeNet-5 on a specific dataset
def train_lenet5(dataset_name, x_train, y_train, x_test, y_test):
    print(f"\nStarting training on {dataset_name} dataset...")
    # Define LeNet-5 model
    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=x_train.shape[1:]))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='tanh'))
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    print(f"\nTraining LeNet-5 on {dataset_name} dataset:")
    model.summary()

    # Train the model
    print(f"Starting training on {dataset_name} dataset...")
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=20,
                        verbose=1,
                        validation_data=(x_test, y_test))
    print(f"Completed training on {dataset_name} dataset.")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy on {dataset_name}: {test_acc}')

    # Predict the classes for the test set
    y_pred = model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
    y_test_classes = tf.argmax(y_test, axis=1).numpy()

    # Calculate precision, recall, F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_classes, y_pred_classes, average='weighted')

    print(f'Precision on {dataset_name}: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')

    # Plot training history
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.title(f'Training and Validation Loss and Accuracy - {dataset_name}')
    plt.legend()
    plt.show()

    return test_acc, precision, recall, f1_score

# Load and preprocess datasets
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_mnist, x_test_mnist = x_train_mnist / 255.0, x_test_mnist / 255.0
x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1)
x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1)
y_train_mnist = to_categorical(y_train_mnist, num_classes=10)
y_test_mnist = to_categorical(y_test_mnist, num_classes=10)

(x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
x_train_fmnist, x_test_fmnist = x_train_fmnist / 255.0, x_test_fmnist / 255.0
x_train_fmnist = x_train_fmnist.reshape(-1, 28, 28, 1)
x_test_fmnist = x_test_fmnist.reshape(-1, 28, 28, 1)
y_train_fmnist = to_categorical(y_train_fmnist, num_classes=10)
y_test_fmnist = to_categorical(y_test_fmnist, num_classes=10)

(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
x_train_cifar, x_test_cifar = x_train_cifar / 255.0, x_test_cifar / 255.0
y_train_cifar = to_categorical(y_train_cifar, num_classes=10)
y_test_cifar = to_categorical(y_test_cifar, num_classes=10)

# Train and evaluate LeNet-5 on each dataset
results = {}

# Train and evaluate on MNIST dataset
acc_mnist, prec_mnist, rec_mnist, f1_mnist = train_lenet5('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)
results['MNIST'] = {'Accuracy': acc_mnist, 'Precision': prec_mnist, 'Recall': rec_mnist, 'F1-score': f1_mnist}

# Train and evaluate on Fashion MNIST dataset
acc_fmnist, prec_fmnist, rec_fmnist, f1_fmnist = train_lenet5('Fashion MNIST', x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist)
results['Fashion MNIST'] = {'Accuracy': acc_fmnist, 'Precision': prec_fmnist, 'Recall': rec_fmnist, 'F1-score': f1_fmnist}

# Train and evaluate on CIFAR-10 dataset
acc_cifar, prec_cifar, rec_cifar, f1_cifar = train_lenet5('CIFAR-10', x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)
results['CIFAR-10'] = {'Accuracy': acc_cifar, 'Precision': prec_cifar, 'Recall': rec_cifar, 'F1-score': f1_cifar}

# Print results
print("\nResults:")
for dataset, metrics in results.items():
    print(f"{dataset}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-score: {metrics['F1-score']:.4f}")
    print()

# Write results to a text file
with open('LeNet-5.txt', 'w') as f:
    f.write("Final Results:\n")
    for dataset, metrics in results.items():
        f.write(f"\n{dataset} dataset:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

# Comparison of metrics
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [metrics['Accuracy'] for metrics in results.values()], label='Accuracy')
plt.bar(results.keys(), [metrics['Precision'] for metrics in results.values()], label='Precision')
plt.bar(results.keys(), [metrics['Recall'] for metrics in results.values()], label='Recall')
plt.bar(results.keys(), [metrics['F1-score'] for metrics in results.values()], label='F1-score')
plt.title('Comparison of Performance Metrics - LeNet-5')
plt.xlabel('Datasets')
plt.ylabel('Metrics')
plt.ylim(0, 1.2)
plt.legend()
plt.show()

print("\nScript execution completed.")