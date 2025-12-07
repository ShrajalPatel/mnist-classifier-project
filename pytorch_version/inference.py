"""
PyTorch Inference Script for MNIST Digit Classification
"""

import torch
import numpy as np
from PIL import Image
import argparse
from model import MNISTNet


def load_model(model_path='pytorch_version/trained_model.pth'):
    """Load the trained PyTorch model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = MNISTNet().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Test accuracy: {checkpoint['test_acc']:.2f}%")
    
    return model, device


def preprocess_image(image_path):
    """
    Preprocess an image for MNIST prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed tensor ready for the model
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize (0-255 to 0-1)
    img_array = img_array / 255.0
    
    # Apply MNIST normalization
    mean, std = 0.1307, 0.3081
    img_array = (img_array - mean) / std
    
    # Flatten to (1, 784)
    img_tensor = torch.FloatTensor(img_array).view(1, -1)
    
    return img_tensor


def predict(model, device, image_tensor):
    """
    Make a prediction on the input image.
    
    Args:
        model: Trained PyTorch model
        device: Device to run inference on
        image_tensor: Preprocessed image tensor
        
    Returns:
        Predicted digit and confidence scores
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        output = model(image_tensor)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get prediction
        confidence, predicted = torch.max(probabilities, 1)
        
        # Convert to numpy
        probs = probabilities.cpu().numpy()[0]
        predicted_digit = predicted.item()
        confidence_score = confidence.item()
    
    return predicted_digit, confidence_score, probs


def main():
    parser = argparse.ArgumentParser(description='MNIST Digit Classification Inference (PyTorch)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='pytorch_version/trained_model.pth',
                        help='Path to the trained model')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, device = load_model(args.model)
    
    # Preprocess image
    print(f"\nPreprocessing image: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Make prediction
    print("Making prediction...")
    predicted_digit, confidence, probabilities = predict(model, device, image_tensor)
    
    # Display results
    print("\n" + "=" * 50)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nProbability Distribution:")
    print("-" * 50)
    for digit in range(10):
        bar = '█' * int(probabilities[digit] * 50)
        print(f"  {digit}: {probabilities[digit]:6.2%} {bar}")
    print("=" * 50)


if __name__ == "__main__":
    main()
