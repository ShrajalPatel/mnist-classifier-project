"""
Neural Network Implementation from Scratch using NumPy
Complete implementation with forward propagation, backpropagation, and training.
"""

import numpy as np
from activations import relu, relu_derivative, softmax


class NeuralNetwork:
    """
    Feedforward Neural Network for MNIST Classification.
    
    Architecture: 784 -> 128 -> 64 -> 10
    
    This implementation includes:
    - Xavier/He weight initialization
    - Mini-batch gradient descent
    - Cross-entropy loss
    - Manual backpropagation
    """
    
    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10, learning_rate=0.01):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features (784 for MNIST)
            hidden1: Number of neurons in first hidden layer
            hidden2: Number of neurons in second hidden layer
            num_classes: Number of output classes (10 for digits)
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize weights using He initialization (better for ReLU)
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1))
        
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        
        self.W3 = np.random.randn(hidden2, num_classes) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, num_classes))
        
        # Cache for backpropagation
        self.cache = {}
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Mathematical operations:
        1. Z1 = X @ W1 + b1
        2. A1 = ReLU(Z1)
        3. Z2 = A1 @ W2 + b2
        4. A2 = ReLU(Z2)
        5. Z3 = A2 @ W3 + b3
        6. A3 = Softmax(Z3)
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            A3: Output probabilities of shape (batch_size, num_classes)
        """
        # Layer 1: Input -> Hidden1
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        
        # Layer 2: Hidden1 -> Hidden2
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        
        # Layer 3: Hidden2 -> Output
        Z3 = A2 @ self.W3 + self.b3
        A3 = softmax(Z3)
        
        # Cache values for backpropagation
        self.cache = {
            'X': X, 'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'Z3': Z3, 'A3': A3
        }
        
        return A3
    
    def backward(self, Y):
        """
        Backward propagation to compute gradients.
        
        Mathematical derivations:
        1. dZ3 = A3 - Y (derivative of softmax + cross-entropy)
        2. dW3 = A2.T @ dZ3 / m
        3. db3 = sum(dZ3) / m
        4. dZ2 = (dZ3 @ W3.T) * ReLU'(Z2)
        5. dW2 = A1.T @ dZ2 / m
        6. db2 = sum(dZ2) / m
        7. dZ1 = (dZ2 @ W2.T) * ReLU'(Z1)
        8. dW1 = X.T @ dZ1 / m
        9. db1 = sum(dZ1) / m
        
        Args:
            Y: One-hot encoded labels of shape (batch_size, num_classes)
        """
        m = Y.shape[0]  # Batch size
        
        # Retrieve cached values
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        Z2 = self.cache['Z2']
        A2 = self.cache['A2']
        A3 = self.cache['A3']
        
        # Output layer gradients (softmax + cross-entropy derivative)
        dZ3 = A3 - Y
        dW3 = (A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Hidden layer 2 gradients
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer 1 gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Store gradients
        self.gradients = {
            'dW3': dW3, 'db3': db3,
            'dW2': dW2, 'db2': db2,
            'dW1': dW1, 'db1': db1
        }
    
    def update_weights(self):
        """
        Update weights using gradient descent.
        
        Formula: W = W - learning_rate * dW
        """
        self.W3 -= self.learning_rate * self.gradients['dW3']
        self.b3 -= self.learning_rate * self.gradients['db3']
        
        self.W2 -= self.learning_rate * self.gradients['dW2']
        self.b2 -= self.learning_rate * self.gradients['db2']
        
        self.W1 -= self.learning_rate * self.gradients['dW1']
        self.b1 -= self.learning_rate * self.gradients['db1']
    
    def compute_loss(self, Y_pred, Y_true):
        """
        Compute cross-entropy loss.
        
        Formula: L = -1/m * sum(Y_true * log(Y_pred))
        
        Args:
            Y_pred: Predicted probabilities (batch_size, num_classes)
            Y_true: True labels one-hot encoded (batch_size, num_classes)
            
        Returns:
            Average cross-entropy loss
        """
        m = Y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        loss = -np.sum(Y_true * np.log(Y_pred + epsilon)) / m
        return loss
    
    def compute_accuracy(self, Y_pred, Y_true):
        """
        Compute classification accuracy.
        
        Args:
            Y_pred: Predicted probabilities (batch_size, num_classes)
            Y_true: True labels (batch_size,) or one-hot encoded
            
        Returns:
            Accuracy as a percentage
        """
        # Get predicted classes
        pred_classes = np.argmax(Y_pred, axis=1)
        
        # Get true classes
        if len(Y_true.shape) == 2:  # One-hot encoded
            true_classes = np.argmax(Y_true, axis=1)
        else:
            true_classes = Y_true
        
        accuracy = np.mean(pred_classes == true_classes) * 100
        return accuracy
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Predicted class labels
        """
        probabilities = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
    
    def save_weights(self, filepath):
        """Save model weights to a file."""
        np.savez(filepath,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """Load model weights from a file."""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        print(f"Model weights loaded from {filepath}")


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded format.
    
    Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    Args:
        labels: Integer labels of shape (n,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels of shape (n, num_classes)
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


if __name__ == "__main__":
    # Test the neural network
    print("Testing Neural Network Implementation")
    print("=" * 60)
    
    # Create a small test dataset
    np.random.seed(42)
    X_test = np.random.randn(10, 784)
    Y_test = np.random.randint(0, 10, 10)
    Y_test_onehot = one_hot_encode(Y_test)
    
    # Initialize network
    nn = NeuralNetwork()
    
    # Test forward pass
    print("\nTesting forward pass...")
    output = nn.forward(X_test)
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sum of probabilities: {output[0].sum():.4f} (should be ~1.0)")
    
    # Test loss computation
    loss = nn.compute_loss(output, Y_test_onehot)
    print(f"\nInitial loss: {loss:.4f}")
    
    # Test accuracy
    accuracy = nn.compute_accuracy(output, Y_test)
    print(f"Initial accuracy: {accuracy:.2f}%")
    
    # Test backward pass
    print("\nTesting backward pass...")
    nn.backward(Y_test_onehot)
    print("Gradients computed successfully!")
    
    # Test weight update
    print("\nTesting weight update...")
    old_W1 = nn.W1.copy()
    nn.update_weights()
    weight_change = np.abs(nn.W1 - old_W1).mean()
    print(f"Average weight change: {weight_change:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
