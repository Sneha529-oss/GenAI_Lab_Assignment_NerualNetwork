# Neural Network from Scratch — Iris Classification

**MIT Academy of Engineering, Alandi, Pune**
Department of CSE (AI/ML) | Course: Generative AI Lab
Practice Lab Assignment 1 — Individual Lab Task

---

## 📌 Overview

This repository contains a **feedforward neural network implemented entirely from scratch in Python** — using only NumPy for the math — with no deep learning libraries (no TensorFlow, PyTorch, or Keras). It covers the full pipeline: forward propagation, backpropagation, and gradient descent, trained on the classic Iris dataset.

| | |
|---|---|
| **Student** | Sneha Chaurasia |
| **PRN** | *(fill in)* |
| **Class / Batch** | T.Y. Tech, *(batch)* |
| **Course** | Generative AI Lab |
| **Dataset** | [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) (UCI ML Repository) |
| **Task** | Multi-class classification (3 classes) |

---

## 🎯 Objective

Implement a simple feedforward neural network from scratch, focusing on:
- Forward pass (weighted sums + activations)
- Backpropagation (gradient derivation via chain rule)
- Training using gradient descent

---

## 📊 Dataset

**Iris Dataset** — [https://archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)

| Property | Value |
|---|---|
| Samples | 150 (50 per class) |
| Features | 4 (sepal length, sepal width, petal length, petal width — all in cm) |
| Classes | 3 (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*) |
| Source | R.A. Fisher (1936), UCI Machine Learning Repository |

---

## 🏗️ Network Architecture

```
Input Layer          Hidden Layer          Output Layer
 (4 neurons)          (8 neurons)           (3 neurons)

   x1 ──┐                ┌── h1 ──┐             ┌── ŷ1 (setosa)
   x2 ──┤                ├── h2 ──┤             │
   x3 ──┼──► [W1, b1] ──►├── h3 ──┼──► [W2,b2]──►├── ŷ2 (versicolor)
   x4 ──┘                ├── ... ─┤             │
                          └── h8 ──┘             └── ŷ3 (virginica)

              ReLU activation          Softmax activation
```

| Layer | Neurons | Activation | Purpose |
|---|---|---|---|
| Input | 4 | — | Raw normalized features |
| Hidden | 8 | ReLU | Non-linear feature extraction |
| Output | 3 | Softmax | Class probability distribution |

**Loss Function:** Categorical Cross-Entropy
**Optimizer:** Batch Gradient Descent
**Learning Rate:** 0.1
**Epochs:** 1000

---

## 🔄 Training Pipeline

```
┌───────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Load & Split │ ──► │  Forward Pass  │ ──► │  Compute Loss     │ ──► │  Backward Pass   │
│  Iris Dataset │     │  (ReLU+Softmax)│     │  (Cross-Entropy)  │     │  (Backprop)      │
└───────────────┘     └────────────────┘     └──────────────────┘     └─────────────────┘
                                                                                │
┌───────────────┐     ┌────────────────┐     ┌──────────────────┐             │
│   Evaluate     │ ◄── │  Repeat for    │ ◄── │  Update Weights   │ ◄──────────┘
│   on Test Set  │     │  N Epochs      │     │  (Gradient Descent)│
└───────────────┘     └────────────────┘     └──────────────────┘
```

---

## 📁 Repository Structure

```
generative-ai-lab-nn-from-scratch/
│
├── README.md                                       ← this file
├── Sneha_Chaurasia_GenAILabAssignment.ipynb         ← main notebook
├── iris.data                                        ← dataset (CSV, no header)
├── requirements.txt                                 ← Python dependencies
│
└── images/
    ├── loss_curve.png                                ← training loss plot
    ├── accuracy_curve.png                             ← training accuracy plot
    └── confusion_matrix.png                           ← test set confusion matrix
```

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd generative-ai-lab-nn-from-scratch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run all cells:
   ```bash
   jupyter notebook Sneha_Chaurasia_GenAILabAssignment.ipynb
   ```
   Ensure `iris.data` is in the same folder as the notebook.

---

## 📦 Requirements

```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Epochs | 1000 |
| Final Training Accuracy | *(fill in after running)* |
| Test Accuracy | *(fill in after running)* |
| Loss Function | Categorical Cross-Entropy |

**Training Loss Curve**
`images/loss_curve.png`

**Training Accuracy Curve**
`images/accuracy_curve.png`

**Confusion Matrix (Test Set)**
`images/confusion_matrix.png`

---

## 🧠 Key Implementation Details

- **Weight Initialization:** He initialization (suited for ReLU activations)
- **Forward Pass:** `Z1 = X·W1 + b1 → A1 = ReLU(Z1) → Z2 = A1·W2 + b2 → A2 = Softmax(Z2)`
- **Backward Pass:** Chain rule applied manually; softmax + cross-entropy gradient simplifies to `dZ2 = A2 - Y`
- **Confusion Matrix:** built manually from scratch using NumPy (no sklearn dependency for this step)

---

## ✅ Submission Checklist

- [x] Code file (Jupyter Notebook)
- [x] Dataset (`iris.data`) / dataset link included above
- [x] Visualizations (loss curve, accuracy curve, confusion matrix)
- [x] Screenshots of model performance metrics (`images/` folder)
- [x] README file

---

## 📝 Declaration

I, **Sneha Chaurasia**, confirm that the work submitted in this repository is my own and has been completed following academic integrity guidelines.

**GitHub Repository Link:** *(https://github.com/Sneha529-oss/GenAI_Lab_Assignment_NerualNetwork)*
