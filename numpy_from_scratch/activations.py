"""
Activation Functions for Neural Network (NumPy Implementation)
All functions implemented from scratch with detailed mathematical explanations.
"""

import numpy as np


def relu(z):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(x) = max(0, x)
    
    Args:
        z: Input array
        
    Returns:
        Activated output
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU activation function.
    
    Formula: f'(x) = 1 if x > 0, else 0
    
    Args:
        z: Input array (pre-activation values)
        
    Returns:
        Gradient of ReLU
    """
    return (z > 0).astype(float)


def softmax(z):
    """
    Softmax activation function for multi-class classification.
    
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    Uses numerical stability trick: subtract max value before exp
    to prevent overflow.
    
    Args:
        z: Input array of shape (batch_size, num_classes)
        
    Returns:
        Probability distribution over classes
    """
    # Numerical stability: subtract max value
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def sigmoid(z):
    """
    Sigmoid activation function.
    
    Formula: σ(x) = 1 / (1 + exp(-x))
    
    Args:
        z: Input array
        
    Returns:
        Activated output between 0 and 1
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow


def sigmoid_derivative(z):
    """
    Derivative of sigmoid activation function.
    
    Formula: σ'(x) = σ(x) * (1 - σ(x))
    
    Args:
        z: Input array (pre-activation values)
        
    Returns:
        Gradient of sigmoid
    """
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    """
    Hyperbolic tangent activation function.
    
    Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        z: Input array
        
    Returns:
        Activated output between -1 and 1
    """
    return np.tanh(z)


def tanh_derivative(z):
    """
    Derivative of tanh activation function.
    
    Formula: tanh'(x) = 1 - tanh²(x)
    
    Args:
        z: Input array (pre-activation values)
        
    Returns:
        Gradient of tanh
    """
    t = tanh(z)
    return 1 - t ** 2


def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    Formula: f(x) = x if x > 0, else alpha * x
    
    Args:
        z: Input array
        alpha: Slope for negative values (default: 0.01)
        
    Returns:
        Activated output
    """
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    """
    Derivative of Leaky ReLU activation function.
    
    Formula: f'(x) = 1 if x > 0, else alpha
    
    Args:
        z: Input array (pre-activation values)
        alpha: Slope for negative values (default: 0.01)
        
    Returns:
        Gradient of Leaky ReLU
    """
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz


# Dictionary for easy access to activation functions
ACTIVATIONS = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'softmax': (softmax, None)  # Softmax derivative handled separately
}


if __name__ == "__main__":
    # Test activation functions
    print("Testing Activation Functions")
    print("=" * 60)
    
    # Test data
    x = np.array([[-2, -1, 0, 1, 2]])
    
    print(f"Input: {x}")
    print(f"\nReLU: {relu(x)}")
    print(f"ReLU Derivative: {relu_derivative(x)}")
    
    print(f"\nSigmoid: {sigmoid(x)}")
    print(f"Sigmoid Derivative: {sigmoid_derivative(x)}")
    
    print(f"\nTanh: {tanh(x)}")
    print(f"Tanh Derivative: {tanh_derivative(x)}")
    
    # Test softmax with multiple classes
    logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
    print(f"\nLogits: {logits}")
    print(f"Softmax: {softmax(logits)}")
    print(f"Sum of softmax (should be 1.0): {softmax(logits).sum(axis=1)}")
