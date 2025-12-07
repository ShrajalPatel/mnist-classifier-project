"""
Training Visualization Script
Generate plots for training/validation loss and accuracy.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def load_training_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def plot_metrics(histories, labels, save_path='public/plots'):
    """
    Plot training metrics for multiple models.
    
    Args:
        histories: List of training history dictionaries
        labels: List of model names
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('MNIST Training Metrics Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Plot 1: Loss over epochs
    for history, label, color in zip(histories, labels, colors):
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'o-', label=f'{label} Train', 
                color=color, linewidth=2, markersize=6)
        ax1.plot(epochs, history['test_loss'], 's--', label=f'{label} Test', 
                color=color, linewidth=2, markersize=6, alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over epochs
    for history, label, color in zip(histories, labels, colors):
        epochs = range(1, len(history['train_acc']) + 1)
        ax2.plot(epochs, history['train_acc'], 'o-', label=f'{label} Train', 
                color=color, linewidth=2, markersize=6)
        ax2.plot(epochs, history['test_acc'], 's--', label=f'{label} Test', 
                color=color, linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([90, 100])  # Focus on high accuracy range
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training comparison plot saved to: {plot_path}")
    
    plt.close()


def plot_individual_model(history, model_name, save_path='public/plots'):
    """
    Plot training metrics for a single model.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} Training Metrics', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', 
            color='#3498db', linewidth=2, markersize=8)
    ax1.plot(epochs, history['test_loss'], 's-', label='Test Loss', 
            color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss over Epochs', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy', 
            color='#3498db', linewidth=2, markersize=8)
    ax2.plot(epochs, history['test_acc'], 's-', label='Test Accuracy', 
            color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy over Epochs', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{model_name.lower().replace(" ", "_")}_training.png'
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"{model_name} training plot saved to: {plot_path}")
    
    plt.close()


def create_summary_table(histories, labels):
    """Print a summary table of final metrics."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Final Train Acc':<18} {'Final Test Acc':<18} {'Best Test Acc':<15}")
    print("-" * 80)
    
    for history, label in zip(histories, labels):
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        best_test_acc = max(history['test_acc'])
        
        print(f"{label:<20} {final_train_acc:>15.2f}% {final_test_acc:>15.2f}% {best_test_acc:>12.2f}%")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics for MNIST models')
    parser.add_argument('--compare', action='store_true', 
                       help='Create comparison plot for all models')
    parser.add_argument('--pytorch', action='store_true',
                       help='Plot PyTorch training metrics')
    parser.add_argument('--tensorflow', action='store_true',
                       help='Plot TensorFlow training metrics')
    parser.add_argument('--numpy', action='store_true',
                       help='Plot NumPy training metrics')
    parser.add_argument('--all', action='store_true',
                       help='Plot all individual models and comparison')
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not any([args.compare, args.pytorch, args.tensorflow, args.numpy, args.all]):
        args.all = True
    
    histories = []
    labels = []
    
    # Load PyTorch history
    if args.pytorch or args.all or args.compare:
        try:
            pytorch_history = load_training_history('pytorch_version/training_history.json')
            histories.append(pytorch_history)
            labels.append('PyTorch')
            if args.pytorch or args.all:
                plot_individual_model(pytorch_history, 'PyTorch')
        except FileNotFoundError:
            print("Warning: PyTorch training history not found. Train the model first.")
    
    # Load TensorFlow history
    if args.tensorflow or args.all or args.compare:
        try:
            tf_history = load_training_history('tensorflow_version/training_history.json')
            histories.append(tf_history)
            labels.append('TensorFlow')
            if args.tensorflow or args.all:
                plot_individual_model(tf_history, 'TensorFlow')
        except FileNotFoundError:
            print("Warning: TensorFlow training history not found. Train the model first.")
    
    # Load NumPy history
    if args.numpy or args.all or args.compare:
        try:
            numpy_history = load_training_history('numpy_from_scratch/training_history.json')
            histories.append(numpy_history)
            labels.append('NumPy')
            if args.numpy or args.all:
                plot_individual_model(numpy_history, 'NumPy')
        except FileNotFoundError:
            print("Warning: NumPy training history not found. Train the model first.")
    
    # Create comparison plot
    if (args.compare or args.all) and len(histories) > 0:
        plot_metrics(histories, labels)
    
    # Print summary table
    if len(histories) > 0:
        create_summary_table(histories, labels)
    else:
        print("No training histories found. Please train at least one model first.")


if __name__ == "__main__":
    main()
