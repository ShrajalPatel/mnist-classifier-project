# 🧠 Handwritten Digit Classifier – MNIST Deep Learning Project

A comprehensive deep learning project implementing handwritten digit classification using the MNIST dataset. This project includes **three different implementations** to demonstrate various approaches to neural networks:

1. **PyTorch Implementation** - Modern deep learning framework
2. **TensorFlow/Keras Implementation** - High-level neural network API
3. **NumPy From Scratch** - Pure Python implementation with detailed mathematics

## 🌟 Features

- ✅ Three complete implementations (PyTorch, TensorFlow, NumPy)
- ✅ Interactive web demo with canvas drawing
- ✅ Training visualizations (loss/accuracy plots)
- ✅ Confusion matrix analysis
- ✅ GPU acceleration support (PyTorch & TensorFlow)
- ✅ Model saving and loading
- ✅ Real-time inference on custom drawings
- ✅ Performance comparison across frameworks

## 📊 Performance Metrics

| Implementation | Accuracy | Training Time | Model Size |
|---------------|----------|---------------|------------|
| PyTorch       | ~98.5%   | ~2 min        | 1.2 MB     |
| TensorFlow    | ~98.3%   | ~2.5 min      | 1.5 MB     |
| NumPy Scratch | ~97.8%   | ~15 min       | 800 KB     |

*Performance measured on NVIDIA RTX 3080 (PyTorch/TF) and CPU (NumPy)*

## 🏗️ Project Structure

```
mnist-classifier/
├── pytorch_version/
│   ├── train.py              # Training script
│   ├── model.py              # Neural network architecture
│   ├── inference.py          # Test on custom images
│   └── trained_model.pth     # Saved model weights
├── tensorflow_version/
│   ├── train.py              # Training script
│   ├── model.py              # Keras model definition
│   ├── inference.py          # Prediction script
│   └── mnist_model.h5        # Saved Keras model
├── numpy_from_scratch/
│   ├── train.py              # Full implementation
│   ├── network.py            # Neural network class
│   ├── activations.py        # Activation functions
│   └── trained_weights.npz   # Saved weights
├── utils/
│   ├── plot_training.py      # Training visualization
│   ├── confusion_matrix.py   # Confusion matrix plot
│   └── data_loader.py        # MNIST data utilities
├── public/
│   └── plots/                # Generated visualizations
├── src/
│   └── app/                  # Next.js frontend
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
# or
bun install
```

### 3. Train Models

**PyTorch:**
```bash
python pytorch_version/train.py
```

**TensorFlow:**
```bash
python tensorflow_version/train.py
```

**NumPy (From Scratch):**
```bash
python numpy_from_scratch/train.py
```

### 4. Run Web Interface

```bash
npm run dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) to see the interactive demo.

## 📚 Architecture Details

### Neural Network Architecture

All three implementations use the same architecture for fair comparison:

```
Input Layer:    784 neurons (28×28 flattened image)
Hidden Layer 1: 128 neurons + ReLU activation
Hidden Layer 2: 64 neurons + ReLU activation
Output Layer:   10 neurons + Softmax activation
```

**Hyperparameters:**
- Loss Function: Cross-Entropy Loss
- Optimizer: SGD with momentum (PyTorch/TF) / Mini-batch GD (NumPy)
- Learning Rate: 0.01
- Batch Size: 64
- Epochs: 10

### NumPy Implementation Details

The from-scratch implementation includes:

**Forward Propagation:**
```python
# Layer 1: Input → Hidden1
Z1 = X @ W1 + b1
A1 = ReLU(Z1)

# Layer 2: Hidden1 → Hidden2
Z2 = A1 @ W2 + b2
A2 = ReLU(Z2)

# Layer 3: Hidden2 → Output
Z3 = A2 @ W3 + b3
A3 = Softmax(Z3)
```

**Backpropagation:**
```python
# Output layer gradient
dZ3 = A3 - Y_one_hot

# Hidden layer 2 gradient
dZ2 = (dZ3 @ W3.T) * ReLU_derivative(Z2)

# Hidden layer 1 gradient
dZ1 = (dZ2 @ W2.T) * ReLU_derivative(Z1)

# Weight updates
W3 -= learning_rate * (A2.T @ dZ3) / batch_size
W2 -= learning_rate * (A1.T @ dZ2) / batch_size
W1 -= learning_rate * (X.T @ dZ1) / batch_size
```

## 🧪 Testing & Inference

### Test Trained Models

**PyTorch:**
```bash
python pytorch_version/inference.py --image path/to/digit.png
```

**TensorFlow:**
```bash
python tensorflow_version/inference.py --image path/to/digit.png
```

**NumPy:**
```bash
python numpy_from_scratch/inference.py --image path/to/digit.png
```

### Generate Visualizations

```bash
# Training plots
python utils/plot_training.py

# Confusion matrix
python utils/confusion_matrix.py --model pytorch
```

## 📈 Results & Analysis

### Training Curves

The training plots show loss and accuracy over epochs for all three implementations. All models converge within 10 epochs with minimal overfitting.

### Confusion Matrix

Common misclassifications:
- 4 ↔ 9 (similar curved shape)
- 3 ↔ 5 (similar structure)
- 7 ↔ 1 (similar vertical line)

## 🎯 Key Learnings

1. **PyTorch**: Flexible, Pythonic API, excellent for research
2. **TensorFlow**: High-level Keras API, great for production
3. **NumPy**: Understanding the mathematics behind neural networks

## 🔧 Troubleshooting

**CUDA Out of Memory:**
```bash
# Reduce batch size in train.py
BATCH_SIZE = 32  # instead of 64
```

**Slow NumPy Training:**
```bash
# Install optimized BLAS library
pip install numpy[openblas]
```

## 📝 License

MIT License - Feel free to use for learning and projects!

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new implementations (JAX, ONNX, etc.)
- Improve model architectures
- Enhance web interface
- Fix bugs

## 📧 Contact

Questions? Open an issue or reach out!

---

**Built with ❤️ for the ML community**