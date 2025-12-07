"""
TensorFlow/Keras Inference Script for MNIST Digit Classification
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import argparse


def load_model(model_path='tensorflow_version/mnist_model.h5'):
    """Load the trained Keras model."""
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


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


def predict(model, image_array):
    """
    Make a prediction on the input image.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        
    Returns:
        Predicted digit, confidence, and probability distribution
    """
    # Make prediction
    probabilities = model.predict(image_array, verbose=0)[0]
    
    # Get predicted class and confidence
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
    parser = argparse.ArgumentParser(description='MNIST Digit Classification Inference (TensorFlow/Keras)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='tensorflow_version/mnist_model.h5',
                        help='Path to the trained model')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Preprocess image
    print(f"\nPreprocessing image: {args.image}")
    image_array = preprocess_image(args.image)
    print(f"Image shape: {image_array.shape}")
    
    # Make prediction
    print("Making prediction...")
    predicted_digit, confidence, probabilities = predict(model, image_array)
    
    # Display results
    print("\n" + "=" * 50)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence * 100:.2f}%")
    visualize_probabilities(probabilities)
    print("=" * 50)


if __name__ == "__main__":
    main()
