"""
TensorFlow/Keras Neural Network Model for MNIST Classification
Architecture: 784 -> 128 -> 64 -> 10
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def create_mnist_model(input_shape=(784,), hidden1=128, hidden2=64, num_classes=10):
    """
    Create a simple feedforward neural network for MNIST digit classification.
    
    Architecture:
        - Input: 784 features (28x28 flattened)
        - Hidden Layer 1: 128 neurons + ReLU
        - Hidden Layer 2: 64 neurons + ReLU
        - Output: 10 classes (digits 0-9) + Softmax
    
    Args:
        input_shape: Shape of input data
        hidden1: Number of neurons in first hidden layer
        hidden2: Number of neurons in second hidden layer
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    
    model = models.Sequential([
        # Input layer (flatten if needed)
        layers.Input(shape=input_shape),
        
        # Hidden Layer 1
        layers.Dense(hidden1, activation='relu', name='hidden1'),
        layers.Dropout(0.2, name='dropout1'),
        
        # Hidden Layer 2
        layers.Dense(hidden2, activation='relu', name='hidden2'),
        layers.Dropout(0.2, name='dropout2'),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='MNIST_Classifier')
    
    return model


def compile_model(model, learning_rate=0.01):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    """
    
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model):
    """Get detailed model summary."""
    model.summary()
    
    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")


if __name__ == "__main__":
    # Test the model
    print("Creating MNIST model...")
    model = create_mnist_model()
    model = compile_model(model)
    
    print("\nModel Architecture:")
    print("=" * 60)
    get_model_summary(model)
    
    # Test forward pass
    import numpy as np
    batch_size = 32
    dummy_input = np.random.randn(batch_size, 784).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    
    print(f"\nTest Forward Pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sum of probabilities (should be ~1.0): {output[0].sum():.4f}")
    print(f"Predicted class for first sample: {np.argmax(output[0])}")
