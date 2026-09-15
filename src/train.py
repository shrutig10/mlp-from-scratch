import numpy as np

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


def evaluate(model, X, y):
    Z1, H1, A = model.forward(X)

    loss = model.loss(A, y)

    predictions = np.argmax(A, axis=1)
    accuracy = np.mean(predictions == y)

    return loss, accuracy

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    for X_batch, y_batch in get_minibatches(
        X_train,
        y_train,
        BATCH_SIZE,
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
            LEARNING_RATE
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

    print( 
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | " 
        f"Train Loss: {train_loss:.4f} | " 
        f"Train Acc: {train_accuracy:.4f} | " 
        f"Val Loss: {val_loss:.4f} | " 
        f"Val Acc: {val_accuracy:.4f}" 
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print("Train losses:", train_losses)
    print("Validation losses:", val_losses)

    print("Train accuracies:", train_accuracies)
    print("Validation accuracies:", val_accuracies)


