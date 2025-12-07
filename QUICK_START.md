# 🚀 Quick Start - MNIST Digit Classifier

Get up and running in 5 minutes!

## Step 1: Install Dependencies (2 min)

```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
npm install
# or
bun install
```

## Step 2: Train a Model (2-15 min)

Choose one implementation to start:

### Option A: PyTorch (Fastest - 2 min on GPU)
```bash
python pytorch_version/train.py
```

### Option B: TensorFlow (Fast - 2.5 min on GPU)
```bash
python tensorflow_version/train.py
```

### Option C: NumPy from Scratch (Educational - 15 min)
```bash
python numpy_from_scratch/train.py
```

## Step 3: Launch Web Demo (30 sec)

```bash
npm run dev
# or
bun dev
```

Open **http://localhost:3000** in your browser!

## Step 4: Test Your Model

```bash
# Test on a custom image
python pytorch_version/inference.py --image your_digit.png
```

## What You Get

✅ **3 Complete Implementations**
- PyTorch: Modern deep learning framework
- TensorFlow: Production-ready models
- NumPy: Learn the math from scratch

✅ **Interactive Web Interface**
- Draw digits with your mouse
- See predictions from all models
- Compare performance metrics

✅ **Comprehensive Tools**
- Training visualization scripts
- Confusion matrix generation
- Command-line inference tools

✅ **Educational Resources**
- Detailed math explanations
- Step-by-step backpropagation
- Architecture diagrams

## Expected Results

| Model | Accuracy | Time |
|-------|----------|------|
| PyTorch | 98.5% | 2 min |
| TensorFlow | 98.3% | 2.5 min |
| NumPy | 97.8% | 15 min |

## Need Help?

📖 **Full Documentation:** See README.md  
🎓 **Training Guide:** See TRAINING_GUIDE.md  
🔍 **Project Overview:** See PROJECT_OVERVIEW.md

## Common Issues

**No GPU?** All models work on CPU (just slower)

**Import errors?** Run: `pip install -r requirements.txt --upgrade`

**Slow training?** Reduce batch size or epochs

---

**That's it! You're ready to explore deep learning! 🎉**
