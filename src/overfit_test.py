import numpy as np
from model import MLP


B = 20
D = 5
H = 10
K = 3

rng = np.random.default_rng(42)

X = rng.normal(size=(B, D))
y = rng.integers(0, K, size=B)

model = MLP(D, H, K, seed=42)

def accuracy(A, y):
    predictions = np.argmax(A, axis=1)
    return np.mean(predictions == y)

learning_rate = 0.1
num_steps = 3000

for step in range(num_steps):
    Z1, H1, A = model.forward(X)
    loss = model.loss(A, y)
    grad_W1, grad_b1, grad_W2, grad_b2 = model.backward(X, y, Z1, H1, A)
    model.update(grad_W1, grad_b1, grad_W2, grad_b2, learning_rate)
    if (step + 1) % 100 == 0:
        print(
            f"Step {step + 1}/{num_steps}, "
            f"Loss: {loss:.4f}, "
            f"Accuracy: {accuracy(A, y):.4f}"
        )