"""
TensorFlow/Keras Training Script for MNIST Digit Classification
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from datetime import datetime
from model import create_mnist_model, compile_model


# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10

# File paths
MODEL_PATH = 'tensorflow_version/mnist_model.h5'
HISTORY_PATH = 'tensorflow_version/training_history.json'


def load_and_prepare_data():
    """Load and preprocess MNIST dataset."""
    
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten images: (28, 28) -> (784,)
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)


def create_callbacks(model_path):
    """Create training callbacks."""
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping (optional, prevents overfitting)
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


def main():
    """Main training function."""
    
    print("=" * 60)
    print("TensorFlow/Keras MNIST Training")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create output directory
    os.makedirs('tensorflow_version', exist_ok=True)
    
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
    
    # Create and compile model
    print("\nCreating model...")
    model = create_mnist_model()
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(MODEL_PATH)
    
    # Training
    print("\n" + "=" * 60)
    print(f"Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    start_time = datetime.now()
    
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nFinal Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    # Save training history
    history_dict = {
        'train_loss': [float(x) for x in history.history['loss']],
        'train_acc': [float(x) * 100 for x in history.history['accuracy']],
        'test_loss': [float(x) for x in history.history['val_loss']],
        'test_acc': [float(x) * 100 for x in history.history['val_accuracy']],
        'training_time': training_time,
        'final_test_acc': float(test_acc) * 100
    }
    
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
