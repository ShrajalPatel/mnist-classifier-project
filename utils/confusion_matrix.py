"""
Confusion Matrix Visualization Script
Generate confusion matrices for trained models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os

# Import model-specific modules
try:
    import torch
    from pytorch_version.model import MNISTNet
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from numpy_from_scratch.network import NeuralNetwork
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def load_mnist_test_data():
    """Load MNIST test dataset."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow required to load MNIST data")
    
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_test = x_test.astype('float32') / 255.0
    
    return x_test, y_test


def get_pytorch_predictions(model_path='pytorch_version/trained_model.pth'):
    """Get predictions from PyTorch model."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping.")
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MNISTNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    x_test, y_test = load_mnist_test_data()
    x_test_flat = x_test.reshape(-1, 784)
    x_test_tensor = torch.FloatTensor(x_test_flat).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(x_test_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return y_test, predictions


def get_tensorflow_predictions(model_path='tensorflow_version/mnist_model.h5'):
    """Get predictions from TensorFlow model."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping.")
        return None, None
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load test data
    x_test, y_test = load_mnist_test_data()
    x_test_flat = x_test.reshape(-1, 784)
    
    # Get predictions
    predictions = model.predict(x_test_flat, verbose=0)
    predictions = np.argmax(predictions, axis=1)
    
    return y_test, predictions


def get_numpy_predictions(model_path='numpy_from_scratch/trained_weights.npz'):
    """Get predictions from NumPy model."""
    if not NUMPY_AVAILABLE:
        print("NumPy network not available. Skipping.")
        return None, None
    
    # Load model
    network = NeuralNetwork()
    network.load_weights(model_path)
    
    # Load test data
    x_test, y_test = load_mnist_test_data()
    x_test_flat = x_test.reshape(-1, 784)
    
    # Get predictions
    predictions = network.predict(x_test_flat)
    
    return y_test, predictions


def plot_confusion_matrix(y_true, y_pred, model_name, save_path='public/plots'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{model_name} Confusion Matrix', fontsize=16, fontweight='bold')
    
    # Plot absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                cbar_kws={'label': 'Count'}, square=True)
    ax1.set_title('Absolute Counts', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    # Plot percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                cbar_kws={'label': 'Percentage (%)'}, square=True)
    ax2.set_title('Normalized (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n{model_name} confusion matrix saved to: {plot_path}")
    
    plt.close()
    
    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print("=" * 70)
    print(classification_report(y_true, y_pred, digits=4))
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("=" * 70)
    
    # Find most common misclassifications
    print(f"\n{model_name} Most Common Misclassifications:")
    print("-" * 70)
    
    misclassifications = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                misclassifications.append((i, j, cm[i, j]))
    
    # Sort by count
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    
    for true_label, pred_label, count in misclassifications[:10]:
        percentage = (count / cm[true_label].sum()) * 100
        print(f"  {true_label} misclassified as {pred_label}: {count} times ({percentage:.2f}%)")
    
    print("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrices for MNIST models')
    parser.add_argument('--model', type=str, choices=['pytorch', 'tensorflow', 'numpy', 'all'],
                       default='all', help='Which model to analyze')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MNIST Confusion Matrix Analysis")
    print("=" * 70)
    
    models_to_analyze = []
    
    if args.model == 'all':
        models_to_analyze = ['pytorch', 'tensorflow', 'numpy']
    else:
        models_to_analyze = [args.model]
    
    for model in models_to_analyze:
        print(f"\nAnalyzing {model.capitalize()} model...")
        print("-" * 70)
        
        try:
            if model == 'pytorch':
                y_true, y_pred = get_pytorch_predictions()
                if y_true is not None and y_pred is not None:
                    plot_confusion_matrix(y_true, y_pred, 'PyTorch')
            
            elif model == 'tensorflow':
                y_true, y_pred = get_tensorflow_predictions()
                if y_true is not None and y_pred is not None:
                    plot_confusion_matrix(y_true, y_pred, 'TensorFlow')
            
            elif model == 'numpy':
                y_true, y_pred = get_numpy_predictions()
                if y_true is not None and y_pred is not None:
                    plot_confusion_matrix(y_true, y_pred, 'NumPy')
        
        except FileNotFoundError as e:
            print(f"Error: Model file not found. Please train the {model} model first.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"Error analyzing {model} model: {e}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
