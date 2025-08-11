# MNIST Handwritten Digit Recognition (CNN)

A complete project to train and deploy a Convolutional Neural Network for MNIST digit classification.

## Files
- `model.py` - model architecture & compile helper
- `train.py` - training script with augmentation & callbacks, saves best model
- `evaluate.py` - evaluate saved model and draw confusion matrix
- `predict.py` - predict digit(s) from image files
- `utils.py` - image loading & preprocessing helpers
- `requirements.txt`

## Setup
1. Create virtualenv and install packages:
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
