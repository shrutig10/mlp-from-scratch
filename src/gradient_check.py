import numpy as np
from model import MLP

B = 3
D = 4
H = 2
K = 3

rng = np.random.default_rng(123)

X = rng.normal(size=(B, D))
y = rng.integers(0, K, size=B)

model = MLP(D, H, K, seed=456)

Z1, H1, A = model.forward(X)

grad_W1, grad_b1, grad_W2, grad_b2 = model.backward(
    X, y, Z1, H1, A
)

def numerical_gradient(model, X, y, param, index, h=1e-5):
    original = param[index]

    param[index] = original + h
    _, _, A = model.forward(X)
    loss_plus = model.loss(A, y)

    param[index] = original - h
    _, _, A = model.forward(X)
    loss_minus = model.loss(A, y)

    param[index] = original

    return (loss_plus - loss_minus) / (2 * h)

def check_gradient(model, X, y, param, analytical_grad, h=1e-5):
    numerical_grad = np.zeros_like(param)

    for index in np.ndindex(param.shape):
        numerical_grad[index] = numerical_gradient(
            model, X, y, param, index, h
        )

    difference = analytical_grad - numerical_grad

    relative_error = (
        np.linalg.norm(difference)
        / max(
            1.0,
            np.linalg.norm(analytical_grad),
            np.linalg.norm(numerical_grad),
        )
    )

    return relative_error

checks = [
    ("W1", model.W1, grad_W1),
    ("b1", model.b1, grad_b1),
    ("W2", model.W2, grad_W2),
    ("b2", model.b2, grad_b2),
]

for name, param, analytical_grad in checks:
    relative_error = check_gradient(
        model, X, y, param, analytical_grad
    )

    print(f"{name}:")
    print("  Analytical norm:", np.linalg.norm(analytical_grad))
    print("  Relative error: ", relative_error)
