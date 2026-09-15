import numpy as np

from data import load_mnist
from model import MLP

DATA_DIR = "data"

B = 20
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10

LEARNING_RATE = 0.1
NUM_STEPS = 3000

SEED = 42

X_train, y_train, X_test, y_test = load_mnist(DATA_DIR)

X = X_train[:B]
y = y_train[:B]

model = MLP(
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    seed=SEED
)

def accuracy(A, y):
    predictions = np.argmax(A, axis=1)
    return np.mean(predictions == y)

for step in range(NUM_STEPS):

    Z1, H1, A = model.forward(X)

    loss = model.loss(A, y)

    grad_W1, grad_b1, grad_W2, grad_b2 = model.backward(
        X,
        y,
        Z1,
        H1,
        A
    )

    model.update(
        grad_W1,
        grad_b1,
        grad_W2,
        grad_b2,
        LEARNING_RATE
    )

    if (step + 1) % 100 == 0:
        print(
            f"Step {step + 1}/{NUM_STEPS} | "
            f"Loss: {loss:.6f} | "
            f"Accuracy: {accuracy(A, y):.4f}"
        )