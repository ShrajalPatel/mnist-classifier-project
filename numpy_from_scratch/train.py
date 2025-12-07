"""
Training Script for NumPy Neural Network from Scratch
Complete MNIST training with mini-batch gradient descent.
"""

import numpy as np
import json
import os
import time
from network import NeuralNetwork, one_hot_encode

# Try to import keras for MNIST data, fallback to manual download
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow not available. Using manual MNIST download...")


# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 64
EPOCHS = 10
HIDDEN1 = 128
HIDDEN2 = 64

# File paths
MODEL_PATH = 'numpy_from_scratch/trained_weights.npz'
HISTORY_PATH = 'numpy_from_scratch/training_history.json'


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    
    if KERAS_AVAILABLE:
        print("Loading MNIST dataset using Keras...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    else:
        print("Please install TensorFlow to download MNIST automatically:")
        print("  pip install tensorflow")
        raise ImportError("TensorFlow required for MNIST dataset")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten images: (28, 28) -> (784,)
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape[1]}")
    print(f"Number of classes: {y_train_onehot.shape[1]}")
    
    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot)


def create_mini_batches(X, Y, batch_size):
    """
    Create mini-batches for training.
    
    Args:
        X: Training data
        Y: Training labels
        batch_size: Size of each mini-batch
        
    Returns:
        List of (X_batch, Y_batch) tuples
    """
    m = X.shape[0]
    mini_batches = []
    
    # Shuffle data
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    
    # Create mini-batches
    num_complete_batches = m // batch_size
    
    for k in range(num_complete_batches):
        X_batch = X_shuffled[k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[k * batch_size:(k + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    # Handle remaining samples
    if m % batch_size != 0:
        X_batch = X_shuffled[num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[num_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches


def train_epoch(network, X_train, Y_train_onehot, batch_size, epoch, total_epochs):
    """
    Train the network for one epoch.
    
    Args:
        network: Neural network instance
        X_train: Training data
        Y_train_onehot: One-hot encoded training labels
        batch_size: Mini-batch size
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        Average loss and accuracy for the epoch
    """
    mini_batches = create_mini_batches(X_train, Y_train_onehot, batch_size)
    
    epoch_loss = 0
    epoch_correct = 0
    total_samples = 0
    
    print(f"\nEpoch {epoch}/{total_epochs}")
    print("-" * 60)
    
    for batch_idx, (X_batch, Y_batch) in enumerate(mini_batches):
        # Forward pass
        Y_pred = network.forward(X_batch)
        
        # Compute loss
        batch_loss = network.compute_loss(Y_pred, Y_batch)
        
        # Backward pass
        network.backward(Y_batch)
        
        # Update weights
        network.update_weights()
        
        # Accumulate metrics
        epoch_loss += batch_loss * X_batch.shape[0]
        
        pred_classes = np.argmax(Y_pred, axis=1)
        true_classes = np.argmax(Y_batch, axis=1)
        epoch_correct += np.sum(pred_classes == true_classes)
        total_samples += X_batch.shape[0]
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            current_acc = (epoch_correct / total_samples) * 100
            print(f"  Batch {batch_idx + 1}/{len(mini_batches)} - "
                  f"Loss: {batch_loss:.4f} - Acc: {current_acc:.2f}%")
    
    avg_loss = epoch_loss / total_samples
    avg_acc = (epoch_correct / total_samples) * 100
    
    return avg_loss, avg_acc


def evaluate(network, X_test, Y_test_onehot):
    """
    Evaluate the network on test data.
    
    Args:
        network: Neural network instance
        X_test: Test data
        Y_test_onehot: One-hot encoded test labels
        
    Returns:
        Test loss and accuracy
    """
    Y_pred = network.forward(X_test)
    test_loss = network.compute_loss(Y_pred, Y_test_onehot)
    test_acc = network.compute_accuracy(Y_pred, Y_test_onehot)
    
    return test_loss, test_acc


def main():
    """Main training function."""
    
    print("=" * 60)
    print("NumPy Neural Network Training from Scratch")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('numpy_from_scratch', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("\nLoading MNIST dataset...")
    (X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot) = load_mnist_data()
    
    # Initialize network
    print(f"\nInitializing neural network...")
    print(f"Architecture: 784 -> {HIDDEN1} -> {HIDDEN2} -> 10")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    network = NeuralNetwork(
        input_size=784,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        num_classes=10,
        learning_rate=LEARNING_RATE
    )
    
    # Count parameters
    total_params = (network.W1.size + network.b1.size +
                   network.W2.size + network.b2.size +
                   network.W3.size + network.b3.size)
    print(f"Total parameters: {total_params:,}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    best_test_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            network, X_train, y_train_onehot, BATCH_SIZE, epoch, EPOCHS
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(network, X_test, y_test_onehot)
        
        epoch_time = time.time() - epoch_start
        
        # Save history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['test_loss'].append(float(test_loss))
        history['test_acc'].append(float(test_acc))
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            network.save_weights(MODEL_PATH)
            print(f"  ✓ Saved best model (Test Acc: {test_acc:.2f}%)")
        
        print("=" * 60)
    
    total_time = time.time() - start_time
    
    # Save training history
    history['training_time'] = total_time
    history['best_test_acc'] = float(best_test_acc)
    
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    print("\nTraining Completed!")
    print("=" * 60)
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print("=" * 60)
    
    # Final evaluation
    print("\nFinal Model Performance:")
    network.load_weights(MODEL_PATH)
    final_loss, final_acc = evaluate(network, X_test, y_test_onehot)
    print(f"  Test Loss: {final_loss:.4f}")
    print(f"  Test Accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()
