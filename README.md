# ASL Alphabet Recognition with CNNs

A deep learning project for recognizing American Sign Language (ASL) fingerspelling alphabet using Convolutional Neural Networks. This project compares traditional computer vision baselines (HOG, LBP) against deep learning approaches, achieving near-perfect classification under controlled conditions while exploring the implications of dataset uniformity on real-world deployment.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Authors](#authors)

---

## Overview

American Sign Language (ASL) is used by approximately 500,000 people in the US and Canada. This project focuses on **static fingerspelling recognition** - classifying hand gestures representing the 26 English letters plus three functional signs (SPACE, DELETE, NOTHING).

**Problem Formulation:**

- **Input:** 96×96 or 128×128 RGB images
- **Output:** Single class label from 29 classes {A-Z, SPACE, DELETE, NOTHING}
- **Task:** Supervised multi-class image classification

**Why This Matters:**

- Creates accessibility tools for ASL learners
- Provides immediate feedback during learning
- Builds foundation for assistive communication technology

---

## Dataset

**Source:** [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/aslalphabet)

**Statistics:**

- 87,000 total RGB images
- 29 classes (26 letters + 3 functional signs)
- Consistent studio lighting and uniform backgrounds
- 200×200 pixel resolution (resized to 96×96 or 128×128 for training)
- **Kaggle Usability Score:** 10/10

**Dataset Split:**

- **Training:** 8,700 images (300 per class)
- **Validation:** 8,700 images (300 per class)
- **Test:** 8,700 images (300 per class)
- **Total Used:** 26,100 images from 87,000 available

**Note:** The dataset exhibits high uniformity - consistent lighting, centered hand positions, and uniform backgrounds. This characteristic significantly impacts model performance and generalization.

---

## Project Structure

```
AI-ASL/
├── data/
│   ├── asl_alphabet_train/          # Training images
│   ├── asl_alphabet_test/           # Test images
│   ├── cache/                       # Feature extraction cache
│   ├── class_indices.json           # Class name mappings
│   ├── asl.py                       # Data loading and splitting logic
│   └── transforms.py                # Image transformations (HOG, augmentations)
│
├── splits/
│   └── asl_val_split_seed42_r10.json  # Train/val/test split indices
│
├── models/
│   ├── cnn_small.py                 # CNN architecture definition
│   ├── logreg_hog.py                # HOG + Logistic Regression baseline
│   └── LBP_reg.py                   # LBP + Logistic Regression baseline
│
├── artifacts/
│   └── asl_runs/                    # Saved models, logs, metrics
│
├── train.py                         # Main CNN training script
├── eval.py                          # Model evaluation script
├── tune.py                          # Hyperparameter tuning script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/NJW04/AI-ASL
cd AI-ASL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place in data/asl_alphabet_train/ and data/asl_alphabet_test/
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
opencv-python>=4.8.0
matplotlib>=3.7.0
optuna>=3.3.0
Pillow>=10.0.0
```

---

## Usage

### 1. Run Baseline Methods

**HOG + Logistic Regression:**

```bash
python models/logreg_hog.py \
    --train-dir data/asl_alphabet_train \
    --test-dir data/asl_alphabet_test \
    --epochs 50 \
    --patience 5
```

**LBP + Logistic Regression:**

```bash
python models/LPB_reg.py \
    --train-dir data/asl_alphabet_train \
    --test-dir data/asl_alphabet_test \
    --epochs 50 \
    --patience 5
```

### 2. Hyperparameter Tuning

```bash
python tune.py \
    --train-dir data/asl_alphabet_train \
    --n-trials 20 \
    --epochs-per-trial 5 \
    --seed 42
```

### 3. Train CNN

**Without Augmentation:**

```bash
python train.py \
    --train-dir data/asl_alphabet_train \
    --lr 0.00192 \
    --weight-decay 0.0000304 \
    --dropout 0.070 \
    --blocks 4 \
    --activation gelu \
    --batch-size 128 \
    --size 128 \
    --epochs 20 \
    --patience 5 \
    --seed 42
```

**With Augmentation:**

```bash
python train.py \
    --train-dir data/asl_alphabet_train \
    --lr 0.00192 \
    --weight-decay 0.0000304 \
    --dropout 0.070 \
    --blocks 4 \
    --activation gelu \
    --batch-size 128 \
    --size 128 \
    --epochs 20 \
    --patience 5 \
    --aug \
    --seed 42
```

### 4. Evaluate Model

```bash
python eval.py \
    --checkpoint artifacts/asl_runs/TIMESTAMP__train-cnn/best.pt \
    --train-dir data/asl_alphabet_train \
    --test-dir data/asl_alphabet_test \
    --size 128
```

**Outputs:**

- `metrics_val.json` / `metrics_test.json` - Performance metrics
- `predictions_val.csv` / `predictions_test.csv` - Per-image predictions with top-3 classes
- `confmat_val.png` / `confmat_test.png` - Confusion matrix visualizations

---

## References

1. **Dataset:**  
   Nagaraj, A. (2018) _ASL Alphabet_. Kaggle. Available at: https://www.kaggle.com/dsv/29550 (Accessed: 28 October 2025).

---

## Authors

- **Benjamin Anton Ruijsch van Dugteren**
- **Nathan Jack Wells**
- **Ryan Benjamin Schapiro**

---

## License

This project is for educational purposes. The dataset is available under Kaggle's academic-use license.

---

## Contact

For questions or collaboration opportunities, please open an issue on GitHub or contact the authors.
