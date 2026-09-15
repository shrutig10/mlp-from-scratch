# MLP from Scratch

Implementation of a multilayer perceptron (MLP) from first principles using NumPy for CS 6301: Deep Learning.

## Project Overview

The implementation includes:

- Affine layers and ReLU activation
- Numerically stable softmax cross-entropy loss
- Manual backpropagation
- Minibatch stochastic gradient descent
- Numerical gradient checking
- Tiny-batch overfitting sanity checks
- Training and validation evaluation
- Controlled experiments on model and training hyperparameters

## Project Structure

```text
mlp-from-scratch/
├── data/                  # MNIST dataset (not tracked by Git)
├── src/
│   ├── model.py           # MLP forward pass, loss, backpropagation
│   ├── data.py            # MNIST loading, splitting, minibatches
│   ├── train.py           # MLP training and evaluation
│   ├── gradient_check.py  # Numerical gradient checking
│   ├── overfit_test.py    # Synthetic-data overfitting sanity check
│   └── mnist_overfit_test.py  # MNIST overfitting sanity check
├── experiments/           # Controlled experiments
├── plots/                 # Generated plots
└── report/                # Final project report
```

## Dataset

This project uses the MNIST handwritten digit dataset.

The dataset files should be placed in the `data/` directory. The `data/` directory is excluded from version control.

## Requirements

* Python 3
* NumPy
* Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the Project

To train the MLP, run from project root:
```bash
python src/train.py
```

To run the numerical gradient check (ensure gradients are being calculated correctly), run from project root:
```bash
python src/gradient_check.py
```

To see results of the MNIST overfitting check (with 20 examples), run from project root:
```bash
python src/mnist_overfit_test.py
```

To see generic overfitting results, run from project root:
```bash
python src/overfit_test.py
```

## Reproducibility

Experiments use fixed random seeds for parameter initialization, data splitting, and minibatch shuffling.

The default seed is `42`.
