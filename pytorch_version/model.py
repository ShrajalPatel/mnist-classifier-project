"""
PyTorch Neural Network Model for MNIST Classification
Architecture: 784 -> 128 -> 64 -> 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Simple feedforward neural network for MNIST digit classification.
    
    Architecture:
        - Input: 784 features (28x28 flattened)
        - Hidden Layer 1: 128 neurons + ReLU
        - Hidden Layer 2: 64 neurons + ReLU
        - Output: 10 classes (digits 0-9) + Softmax
    """
    
    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        # Dropout for regularization (optional, can help prevent overfitting)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 784)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Flatten the input if it's still 2D (28x28)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Layer 1: Input -> Hidden1 with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2: Hidden1 -> Hidden2 with ReLU activation
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3: Hidden2 -> Output (logits, no activation)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Make predictions with softmax probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class probabilities
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = MNISTNet()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 784)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    probs = model.predict(dummy_input)
    print(f"Prediction probabilities shape: {probs.shape}")
    print(f"Sum of probabilities (should be ~1.0): {probs[0].sum().item():.4f}")
