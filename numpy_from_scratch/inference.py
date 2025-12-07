"""
Inference Script for NumPy Neural Network
Test trained model on custom images.
"""

import numpy as np
from PIL import Image
import argparse
from network import NeuralNetwork


def load_model(model_path='numpy_from_scratch/trained_weights.npz'):
    """Load the trained NumPy model."""
    network = NeuralNetwork()
    network.load_weights(model_path)
    print(f"Model loaded from {model_path}")
    return network


def preprocess_image(image_path):
    """
    Preprocess an image for MNIST prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed numpy array ready for the model
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize (0-255 to 0-1)
    img_array = img_array / 255.0
    
    # Flatten to (1, 784)
    img_array = img_array.reshape(1, -1)
    
    return img_array


def predict(network, image_array):
    """
    Make a prediction on the input image.
    
    Args:
        network: Trained neural network
        image_array: Preprocessed image array
        
    Returns:
        Predicted digit, confidence, and probability distribution
    """
    # Forward pass
    probabilities = network.forward(image_array)[0]
    
    # Get prediction
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit]
    
    return predicted_digit, confidence, probabilities


def visualize_probabilities(probabilities):
    """Print a visual representation of probability distribution."""
    print("\nProbability Distribution:")
    print("-" * 50)
    for digit in range(10):
        bar = '█' * int(probabilities[digit] * 50)
        print(f"  {digit}: {probabilities[digit]:6.2%} {bar}")


def main():
    parser = argparse.ArgumentParser(description='MNIST Digit Classification Inference (NumPy from Scratch)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='numpy_from_scratch/trained_weights.npz',
                        help='Path to the trained model weights')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    network = load_model(args.model)
    
    # Preprocess image
    print(f"\nPreprocessing image: {args.image}")
    image_array = preprocess_image(args.image)
    print(f"Image shape: {image_array.shape}")
    
    # Make prediction
    print("Making prediction...")
    predicted_digit, confidence, probabilities = predict(network, image_array)
    
    # Display results
    print("\n" + "=" * 50)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence * 100:.2f}%")
    visualize_probabilities(probabilities)
    print("=" * 50)


if __name__ == "__main__":
    main()
