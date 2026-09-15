import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))

from model import MLP
from train import train, evaluate, plot_metrics
from data import load_mnist, train_val_split


DATA_DIR = "data"

INPUT_DIM = 784
HIDDEN_DIM = 256
OUTPUT_DIM = 10

BATCH_SIZE = 16
LEARNING_RATE = 0.1
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
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    seed=SEED
)

rng = np.random.default_rng(SEED)

train_losses, val_losses, train_accuracies, val_accuracies = train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    rng=rng
)

plot_metrics(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    prefix='final_'
)

test_loss, test_accuracy = evaluate(model, X_test, y_test)
print("Final test loss:", test_loss)
print("Final test accuracy:", test_accuracy)