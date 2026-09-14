import numpy as np

# initializing the parameters for a simple neural network --> simple example before MNIST

import numpy as np


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / input_dim),
            size=(hidden_dim, input_dim)
        )
        self.b1 = np.zeros(hidden_dim)

        self.W2 = rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / hidden_dim),
            size=(output_dim, hidden_dim)
        )
        self.b2 = np.zeros(output_dim)


    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H1 = np.maximum(0, Z1)

        A = H1 @ self.W2.T + self.b2

        return Z1, H1, A


    def loss(self, A, y):
        m = np.max(A, axis=1, keepdims=True)
        shifted = A - m

        logsumexp = m + np.log(
            np.sum(np.exp(shifted), axis=1, keepdims=True)
        )

        logsumexp = logsumexp.ravel()

        correct_logits = A[np.arange(A.shape[0]), y]

        losses = logsumexp - correct_logits

        return np.mean(losses)


    def backward(self, X, y, Z1, H1, A):
        B = X.shape[0]

        # Compute the softmax probabilities
        m = np.max(A, axis=1, keepdims=True)
        shifted = A - m

        exp_shifted = np.exp(shifted)
        P = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # One hot encoding of labels
        T = np.zeros_like(P)
        T[np.arange(B), y] = 1

        # Output layer error
        delta2 = (P - T) / B

        # Output layer gradients
        grad_W2 = delta2.T @ H1
        grad_b2 = np.sum(delta2, axis=0)

        # Hidden layer error; propogate through the weights of the output layer
        grad_H1 = delta2 @ self.W2

        # Backprop through ReLU activation
        relu_grad = (Z1 > 0).astype(float)
        delta1 = grad_H1 * relu_grad

        # First layer gradients
        grad_W1 = delta1.T @ X
        grad_b1 = np.sum(delta1, axis=0)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def update(self, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate):
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1

        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2

if __name__ == "__main__":
    B = 4
    D = 5
    H = 3
    K = 2

    rng = np.random.default_rng(42)

    X = rng.normal(size=(B, D))
    y = rng.integers(0, K, size=B)

    model = MLP(D, H, K)

    Z1, H1, A = model.forward(X)

    loss = model.loss(A, y)
    print(loss)
 
    grad_W1, grad_b1, grad_W2, grad_b2 = model.backward(
        X, y, Z1, H1, A
    )

    model.update(grad_W1, grad_b1, grad_W2, grad_b2, learning_rate=0.01)

    Z1, H1, A = model.forward(X)

    loss = model.loss(A, y)
    print(loss)