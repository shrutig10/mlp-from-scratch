import numpy as np
import matplotlib.pyplot as plt

from data import load_mnist, train_val_split, get_minibatches
from model import MLP

DATA_DIR = "data"

INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10

BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 5

SEED = 42


def evaluate(model, X, y):
    Z1, H1, A = model.forward(X)

    loss = model.loss(A, y)

    predictions = np.argmax(A, axis=1)
    accuracy = np.mean(predictions == y)

    return loss, accuracy


def train_one_epoch(model, X_train, y_train, batch_size, learning_rate, rng):
    for X_batch, y_batch in get_minibatches(
        X_train,
        y_train,
        batch_size,
        rng
    ):
        Z1, H1, A = model.forward(X_batch)

        loss = model.loss(A, y_batch)

        grad_W1, grad_b1, grad_W2, grad_b2 = model.backward(
            X_batch,
            y_batch,
            Z1,
            H1,
            A
        )

        model.update(
            grad_W1,
            grad_b1,
            grad_W2,
            grad_b2,
            learning_rate
        )


def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    learning_rate,
    num_epochs,
    rng
):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_one_epoch(
            model,
            X_train,
            y_train,
            batch_size,
            learning_rate,
            rng
        )

        train_loss, train_accuracy = evaluate(
            model,
            X_train,
            y_train
        )

        val_loss, val_accuracy = evaluate(
            model,
            X_val,
            y_val
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    return (
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies
    )


def plot_metrics(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    prefix="",
):
    epochs = range(1, len(train_losses) + 1)

    # Loss Curve
    plt.figure()

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig(f"plots/{prefix}loss_curve.png")
    plt.show()

    # Accuracy Curve
    plt.figure()

    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid(True)

    plt.savefig(f"plots/{prefix}accuracy_curve.png")
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist(DATA_DIR)

    X_train, y_train, X_val, y_val = train_val_split(
        X_train,
        y_train,
        val_size=10000,
        seed=SEED
    )

    model = MLP(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        seed=SEED
    )

    rng = np.random.default_rng(SEED)

    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        BATCH_SIZE,
        LEARNING_RATE,
        NUM_EPOCHS,
        rng
    )
    plot_metrics(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies
    )