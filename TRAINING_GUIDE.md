# Complete Training Guide - MNIST Digit Classifier

## 🎯 Training Workflow

### Phase 1: Environment Setup (5 minutes)

#### 1.1 Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Verify Installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
```

#### 1.2 Check GPU Availability
```bash
# PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# TensorFlow GPU
python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
```

### Phase 2: Train PyTorch Model (2-10 minutes)

#### 2.1 Start Training
```bash
python pytorch_version/train.py
```

**Expected Output:**
```
Using device: cuda
PyTorch version: 2.x.x

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Model architecture:
MNISTNet(
  (fc1): Linear(in_features=784, out_features=128)
  (fc2): Linear(in_features=128, out_features=64)
  (fc3): Linear(in_features=64, out_features=10)
  (dropout): Dropout(p=0.2)
)

Starting training for 10 epochs...
============================================================
Epoch 1/10: 100%|████████████| 938/938 [00:12<00:00, loss: 0.2841, acc: 91.85%]

Epoch 1/10:
  Train Loss: 0.3234 | Train Acc: 90.56%
  Test Loss:  0.1523 | Test Acc:  95.42%
  ✓ Saved best model (Test Acc: 95.42%)
------------------------------------------------------------
...
```

#### 2.2 Monitor Training Progress
- Watch for decreasing loss
- Look for increasing accuracy
- Check for overfitting (train acc >> test acc)

#### 2.3 Training Complete
```
Training completed!
Best test accuracy: 98.52%
Model saved to: pytorch_version/trained_model.pth
History saved to: pytorch_version/training_history.json
```

### Phase 3: Train TensorFlow Model (2.5-12 minutes)

#### 3.1 Start Training
```bash
python tensorflow_version/train.py
```

**Expected Output:**
```
============================================================
TensorFlow/Keras MNIST Training
============================================================
TensorFlow version: 2.x.x
GPU available: True

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Creating model...

Model Architecture:
Model: "MNIST_Classifier"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
hidden1 (Dense)             (None, 128)               100480    
dropout1 (Dropout)          (None, 128)               0         
hidden2 (Dense)             (None, 64)                8256      
dropout2 (Dropout)          (None, 64)                0         
output (Dense)              (None, 10)                650       
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0

Starting training for 10 epochs...
============================================================
Epoch 1/10
938/938 [==============================] - 3s 3ms/step
    - loss: 0.3145 - accuracy: 0.9078 - val_loss: 0.1456 - val_accuracy: 0.9565
...
```

### Phase 4: Train NumPy Model (15-30 minutes)

#### 4.1 Start Training
```bash
python numpy_from_scratch/train.py
```

**Expected Output:**
```
============================================================
NumPy Neural Network Training from Scratch
============================================================

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Initializing neural network...
Architecture: 784 -> 128 -> 64 -> 10
Learning rate: 0.01
Batch size: 64
Total parameters: 109,386

Starting training for 10 epochs...
============================================================

Epoch 1/10
------------------------------------------------------------
  Batch 100/938 - Loss: 0.4523 - Acc: 87.34%
  Batch 200/938 - Loss: 0.3214 - Acc: 90.12%
  ...

Epoch 1/10 Summary:
  Train Loss: 0.3456 | Train Acc: 89.67%
  Test Loss:  0.1876 | Test Acc:  94.23%
  Time: 92.34s
  ✓ Saved best model (Test Acc: 94.23%)
============================================================
...
```

### Phase 5: Generate Visualizations

#### 5.1 Create Training Plots
```bash
python utils/plot_training.py --all
```

**Generates:**
- `public/plots/training_comparison.png` - All models compared
- `public/plots/pytorch_training.png` - PyTorch metrics
- `public/plots/tensorflow_training.png` - TensorFlow metrics
- `public/plots/numpy_training.png` - NumPy metrics

#### 5.2 Generate Confusion Matrices
```bash
python utils/confusion_matrix.py --model all
```

**Generates:**
- `public/plots/pytorch_confusion_matrix.png`
- `public/plots/tensorflow_confusion_matrix.png`
- `public/plots/numpy_confusion_matrix.png`

**Console Output:**
```
============================================================
PyTorch Classification Report:
============================================================
              precision    recall  f1-score   support

           0     0.9878    0.9898    0.9888       980
           1     0.9912    0.9947    0.9929      1135
           2     0.9835    0.9845    0.9840      1032
           3     0.9821    0.9822    0.9821      1010
           4     0.9837    0.9837    0.9837       982
           5     0.9776    0.9777    0.9777       892
           6     0.9854    0.9854    0.9854       958
           7     0.9805    0.9815    0.9810      1028
           8     0.9763    0.9723    0.9743       974
           9     0.9741    0.9741    0.9741      1009

    accuracy                         0.9852     10000
   macro avg     0.9822    0.9826    0.9824     10000
weighted avg     0.9852    0.9852    0.9852     10000

Overall Accuracy: 98.52%
============================================================

PyTorch Most Common Misclassifications:
----------------------------------------------------------------------
  4 misclassified as 9: 12 times (1.22%)
  9 misclassified as 4: 10 times (0.99%)
  3 misclassified as 5: 9 times (0.89%)
  5 misclassified as 3: 8 times (0.90%)
  7 misclassified as 1: 7 times (0.68%)
...
```

### Phase 6: Test Inference

#### 6.1 Test with Sample Images
```bash
# PyTorch
python pytorch_version/inference.py --image test_digit.png

# Expected output:
==================================================
Predicted Digit: 7
Confidence: 99.23%

Probability Distribution:
--------------------------------------------------
  0:  0.01% 
  1:  0.05% ▌
  2:  0.12% █
  3:  0.08% ▌
  4:  0.34% ███
  5:  0.06% ▌
  6:  0.02% 
  7: 99.23% ██████████████████████████████████████████████████
  8:  0.07% ▌
  9:  0.02% 
==================================================
```

## 📊 Understanding Training Metrics

### Loss
- **Cross-Entropy Loss:** Measures prediction confidence
- **Good Values:** < 0.1 after training
- **Warning Signs:** 
  - Loss increasing: Learning rate too high
  - Loss stuck: Learning rate too low or vanishing gradients

### Accuracy
- **Training Accuracy:** How well model fits training data
- **Test Accuracy:** True model performance (more important)
- **Good Values:** > 97% for MNIST
- **Warning Signs:**
  - Train acc >> Test acc: Overfitting
  - Both low: Underfitting

### Training Time
- **PyTorch GPU:** 1-2 minutes
- **TensorFlow GPU:** 2-3 minutes
- **NumPy CPU:** 15-30 minutes

## 🐛 Troubleshooting

### Problem: CUDA Out of Memory
**Solution:**
```python
# Edit train.py
BATCH_SIZE = 32  # Reduce from 64
```

### Problem: Training Too Slow (CPU)
**Solution:**
```bash
# Install optimized libraries
pip install numpy[openblas]

# Or use fewer epochs
python pytorch_version/train.py  # Edit EPOCHS = 5
```

### Problem: Low Accuracy (< 95%)
**Possible Causes:**
1. Learning rate too high/low
2. Insufficient training epochs
3. Data normalization issues

**Solutions:**
```python
# Try different learning rates
LEARNING_RATE = 0.001  # Smaller
LEARNING_RATE = 0.1    # Larger

# More epochs
EPOCHS = 15

# Verify data normalization
print(f"Data range: {x_train.min()} to {x_train.max()}")
# Should be 0.0 to 1.0
```

### Problem: Overfitting
**Symptoms:**
- Train acc: 99%+
- Test acc: < 95%

**Solutions:**
```python
# Increase dropout
self.dropout = nn.Dropout(0.3)  # From 0.2

# Add regularization
optimizer = optim.SGD(model.parameters(), lr=0.01, 
                      momentum=0.9, weight_decay=1e-4)
```

## 🎓 Training Best Practices

### 1. Monitor Training
- Watch both train and test metrics
- Look for smooth convergence
- Check for overfitting early

### 2. Save Checkpoints
- Best model based on test accuracy
- Periodic checkpoints every N epochs
- Save optimizer state for resuming

### 3. Experiment Tracking
- Log hyperparameters
- Save training history
- Compare multiple runs

### 4. Validation
- Use separate validation set (optional)
- Cross-validation for small datasets
- Test on real-world images

## 📈 Advanced Training Techniques

### Data Augmentation
```python
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)
```

### Early Stopping
```python
if epochs_without_improvement > patience:
    print("Early stopping triggered")
    break
```

## ✅ Training Checklist

- [ ] Install all dependencies
- [ ] Verify GPU availability (optional)
- [ ] Train PyTorch model
- [ ] Train TensorFlow model
- [ ] Train NumPy model
- [ ] Generate training plots
- [ ] Create confusion matrices
- [ ] Test inference on sample images
- [ ] Review model performance
- [ ] Save all artifacts

## 🎉 Success Criteria

Your training is successful if:
- ✅ All three models train without errors
- ✅ Test accuracy > 97% for all models
- ✅ Training plots show smooth convergence
- ✅ Inference works on custom images
- ✅ Confusion matrices generated
- ✅ Models saved successfully

---

**Need Help?**
- Review error messages carefully
- Check file paths and dependencies
- Verify data downloaded correctly
- Try reducing batch size or epochs
- Consult the main README.md

**Happy Training! 🚀**
