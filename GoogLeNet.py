import numpy as np
import datetime
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to build GoogLeNet (Inception v1) model
def build_googlenet(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    conv1_7x7_s2 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1_7x7_s2')(input_layer)
    maxpool1_3x3_s2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1_3x3_s2')(conv1_7x7_s2)
    
    conv2_3x3_reduce = layers.Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2_3x3_reduce')(maxpool1_3x3_s2)
    conv2_3x3 = layers.Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_3x3')(conv2_3x3_reduce)
    maxpool2_3x3_s2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool2_3x3_s2')(conv2_3x3)
    
    inception_3a = inception_module(maxpool2_3x3_s2, 64, 96, 128, 16, 32, 32)
    inception_3b = inception_module(inception_3a, 128, 128, 192, 32, 96, 64)
    maxpool3_3x3_s2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool3_3x3_s2')(inception_3b)
    
    inception_4a = inception_module(maxpool3_3x3_s2, 192, 96, 208, 16, 48, 64)
    inception_4b = inception_module(inception_4a, 160, 112, 224, 24, 64, 64)
    inception_4c = inception_module(inception_4b, 128, 128, 256, 24, 64, 64)
    inception_4d = inception_module(inception_4c, 112, 144, 288, 32, 64, 64)
    inception_4e = inception_module(inception_4d, 256, 160, 320, 32, 128, 128)
    maxpool4_3x3_s2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool4_3x3_s2')(inception_4e)
    
    inception_5a = inception_module(maxpool4_3x3_s2, 256, 160, 320, 32, 128, 128)
    inception_5b = inception_module(inception_5a, 384, 192, 384, 48, 128, 128)
    
    # Adjust avgpool pool_size to be compatible with different input sizes
    if input_shape[0] == 28 and input_shape[1] == 28:
        avgpool = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid', name='avgpool')(inception_5b)
    elif input_shape[0] == 32 and input_shape[1] == 32:
        avgpool = layers.AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid', name='avgpool')(inception_5b)
    
    drop = layers.Dropout(0.4)(avgpool)
    
    flat = layers.Flatten()(drop)
    output = layers.Dense(num_classes, activation='softmax', name='output')(flat)
    
    model = models.Model(inputs=input_layer, outputs=output, name='inception_v1')
    
    return model

def inception_module(prev_layer, conv1_filters, conv3_reduce_filters, conv3_filters, conv5_reduce_filters, conv5_filters, pool_filters):
    conv1x1 = layers.Conv2D(conv1_filters, (1, 1), padding='same', activation='relu')(prev_layer)
    
    conv3x3_reduce = layers.Conv2D(conv3_reduce_filters, (1, 1), padding='same', activation='relu')(prev_layer)
    conv3x3 = layers.Conv2D(conv3_filters, (3, 3), padding='same', activation='relu')(conv3x3_reduce)
    
    conv5x5_reduce = layers.Conv2D(conv5_reduce_filters, (1, 1), padding='same', activation='relu')(prev_layer)
    conv5x5 = layers.Conv2D(conv5_filters, (5, 5), padding='same', activation='relu')(conv5x5_reduce)
    
    pool_proj = layers.Conv2D(pool_filters, (1, 1), padding='same', activation='relu')(layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(prev_layer))
    
    return layers.concatenate([conv1x1, conv3x3, conv5x5, pool_proj])

# Function to preprocess and load data
def load_and_preprocess_data():
    from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
    
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
    
    # Normalize data
    x_train_mnist = x_train_mnist.astype('float32') / 255.0
    x_test_mnist = x_test_mnist.astype('float32') / 255.0
    x_train_fmnist = x_train_fmnist.astype('float32') / 255.0
    x_test_fmnist = x_test_fmnist.astype('float32') / 255.0
    x_train_cifar = x_train_cifar.astype('float32') / 255.0
    x_test_cifar = x_test_cifar.astype('float32') / 255.0
    
    # Reshape for CNN input
    x_train_mnist = np.expand_dims(x_train_mnist, axis=-1)
    x_test_mnist = np.expand_dims(x_test_mnist, axis=-1)
    x_train_fmnist = np.expand_dims(x_train_fmnist, axis=-1)
    x_test_fmnist = np.expand_dims(x_test_fmnist, axis=-1)
    
    return (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist), \
           (x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist), \
           (x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)

# Function to train GoogLeNet on a dataset
def train_googlenet(dataset_name, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    # Build GoogLeNet model
    model = build_googlenet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print(f"Training GoogLeNet on {dataset_name} dataset...")
    history = model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=128, validation_data=(x_test, to_categorical(y_test)), verbose=1)
    
    # Evaluate the model
    print(f"Evaluating GoogLeNet on {dataset_name} dataset...")
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
                   f"F1 score: {f1:.4f}\n"
    
    print(metrics_text)
    
    with open(f'{dataset_name}_metrics.txt', 'w') as f:
        f.write(metrics_text)    
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} - GoogLeNet Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} - GoogLeNet Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_googlenet_training_history.png")
    plt.show()

# Load and preprocess the datasets
(x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist), \
(x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist), \
(x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar) = load_and_preprocess_data()

# Train and evaluate the model on each dataset
train_googlenet('MNIST', x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)
train_googlenet('Fashion_MNIST', x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist)
train_googlenet('CIFAR-10', x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar)