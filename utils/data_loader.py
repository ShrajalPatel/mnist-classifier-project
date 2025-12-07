"""
MNIST Data Utilities
Helper functions for loading and preprocessing MNIST data.
"""

import numpy as np

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


def load_mnist():
    """
    Load MNIST dataset.
    
    Returns:
        (x_train, y_train), (x_test, y_test): Training and test data
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required to load MNIST dataset.\n"
                         "Install with: pip install tensorflow")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"MNIST Dataset Loaded:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Image shape: {x_train.shape[1:]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)


def normalize_images(images):
    """
    Normalize pixel values to [0, 1].
    
    Args:
        images: Array of images with pixel values in [0, 255]
        
    Returns:
        Normalized images
    """
    return images.astype('float32') / 255.0


def flatten_images(images):
    """
    Flatten 2D images to 1D vectors.
    
    Args:
        images: Array of shape (n, height, width)
        
    Returns:
        Flattened images of shape (n, height*width)
    """
    return images.reshape(images.shape[0], -1)


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
        labels: Integer labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def preprocess_for_training(images, labels, flatten=True, one_hot=True):
    """
    Complete preprocessing pipeline for training.
    
    Args:
        images: Raw images
        labels: Integer labels
        flatten: Whether to flatten images
        one_hot: Whether to one-hot encode labels
        
    Returns:
        Preprocessed images and labels
    """
    # Normalize
    images = normalize_images(images)
    
    # Flatten if needed
    if flatten:
        images = flatten_images(images)
    
    # One-hot encode if needed
    if one_hot:
        labels = one_hot_encode(labels)
    
    return images, labels


def get_sample_images(x_data, y_data, num_samples=10, digit=None):
    """
    Get sample images from the dataset.
    
    Args:
        x_data: Image data
        y_data: Labels
        num_samples: Number of samples to return
        digit: Specific digit to sample (None for random)
        
    Returns:
        Sample images and labels
    """
    if digit is not None:
        # Get samples of specific digit
        indices = np.where(y_data == digit)[0]
        selected_indices = np.random.choice(indices, num_samples, replace=False)
    else:
        # Get random samples
        selected_indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    return x_data[selected_indices], y_data[selected_indices]


def create_mini_batches(x_data, y_data, batch_size=64, shuffle=True):
    """
    Create mini-batches for training.
    
    Args:
        x_data: Training data
        y_data: Training labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        
    Returns:
        List of (x_batch, y_batch) tuples
    """
    m = x_data.shape[0]
    mini_batches = []
    
    if shuffle:
        permutation = np.random.permutation(m)
        x_shuffled = x_data[permutation]
        y_shuffled = y_data[permutation]
    else:
        x_shuffled = x_data
        y_shuffled = y_data
    
    num_complete_batches = m // batch_size
    
    for k in range(num_complete_batches):
        x_batch = x_shuffled[k * batch_size:(k + 1) * batch_size]
        y_batch = y_shuffled[k * batch_size:(k + 1) * batch_size]
        mini_batches.append((x_batch, y_batch))
    
    # Handle remaining samples
    if m % batch_size != 0:
        x_batch = x_shuffled[num_complete_batches * batch_size:]
        y_batch = y_shuffled[num_complete_batches * batch_size:]
        mini_batches.append((x_batch, y_batch))
    
    return mini_batches


def visualize_samples(x_data, y_data, num_samples=25):
    """
    Visualize sample images from the dataset.
    
    Args:
        x_data: Image data (can be flattened or 2D)
        y_data: Labels
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt
    
    # Get random samples
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    # Reshape if flattened
    if len(x_data.shape) == 2 and x_data.shape[1] == 784:
        images = x_data[indices].reshape(-1, 28, 28)
    else:
        images = x_data[indices]
    
    labels = y_data[indices]
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}', fontsize=10)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('public/plots/mnist_samples.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved to: public/plots/mnist_samples.png")
    plt.close()


def get_dataset_statistics(x_data, y_data):
    """
    Print dataset statistics.
    
    Args:
        x_data: Image data
        y_data: Labels
    """
    print("\nDataset Statistics:")
    print("=" * 60)
    print(f"Number of samples: {len(x_data)}")
    print(f"Image shape: {x_data.shape[1:]}")
    print(f"Data type: {x_data.dtype}")
    print(f"Value range: [{x_data.min():.4f}, {x_data.max():.4f}]")
    print(f"Mean: {x_data.mean():.4f}")
    print(f"Std: {x_data.std():.4f}")
    
    print("\nClass Distribution:")
    print("-" * 60)
    unique, counts = np.unique(y_data, return_counts=True)
    for digit, count in zip(unique, counts):
        percentage = (count / len(y_data)) * 100
        print(f"  Digit {digit}: {count:5d} samples ({percentage:5.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test data loading and utilities
    print("Testing MNIST Data Utilities")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Get statistics
    get_dataset_statistics(x_train, y_train)
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    x_processed, y_processed = preprocess_for_training(x_train[:100], y_train[:100])
    print(f"Processed shape: {x_processed.shape}")
    print(f"Label shape: {y_processed.shape}")
    
    # Test mini-batch creation
    print("\nTesting mini-batch creation...")
    batches = create_mini_batches(x_processed, y_processed, batch_size=32)
    print(f"Number of batches: {len(batches)}")
    print(f"First batch shape: {batches[0][0].shape}")
    
    # Visualize samples
    print("\nCreating sample visualization...")
    visualize_samples(x_train, y_train, num_samples=25)
    
    print("\nAll tests passed! ✓")
