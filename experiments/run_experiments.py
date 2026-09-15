import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))

from data import load_mnist, train_val_split
from model import MLP
from train import train

import numpy as np

DATA_DIR = "data"
SEED = 42

X_train, y_train, X_test, y_test = load_mnist(DATA_DIR)

X_train, y_train, X_val, y_val = train_val_split(
    X_train,
    y_train,
    val_size=10000,
    seed=SEED
)

def run_experiment(
    hidden_dim,
    batch_size,
    learning_rate,
    seed
):
    model = MLP(
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10,
        seed=seed
    )

    rng = np.random.default_rng(seed)

    _, val_losses, _, val_accuracies = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=5,
        rng=rng
    )

    return val_losses, val_accuracies

if __name__ == "__main__":
    # Experiment 1: Learning rate
    # Tested: [0.001, 0.01, 0.1]
    # Results:
    # 0.001 -> 0.8693 ± 0.0019
    # 0.01  -> 0.9289 ± 0.0005
    # 0.1   -> 0.9710 ± 0.0003

    # Experiment 2: Hidden dimension
    # Tested: [32, 128, 256]
    # Results:
    # 32  -> 0.9576 ± 0.0007
    # 128 -> 0.9710 ± 0.0003
    # 256 -> 0.9729 ± 0.0014

    # Experiment 3: Batch size
    # Tested: [16, 32, 128]
    # Results:
    # 16  -> 0.9735 ± 0.0006
    # 32  -> 0.9710 ± 0.0003
    # 128 -> 0.9508 ± 0.0017
    # Currently active

    learning_rates = [0.001, 0.01, 0.1]
    hidden_dims = [32, 128, 256]
    batch_sizes = [16, 32, 128]
    seeds = [42, 123, 456]

    results = []

    for batch_size in batch_sizes:
        accuracies = []

        for seed in seeds:
            _, val_accuracies = run_experiment(
                hidden_dim=128,
                batch_size=batch_size,
                learning_rate=0.1,
                seed=seed
            )

            final_accuracy = val_accuracies[-1]
            accuracies.append(final_accuracy)

            print(
                f"Batch size: {batch_size}, "
                f"Seed: {seed}, "
                f"Validation accuracy: {final_accuracy:.4f}"
            )

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        results.append((batch_size, mean_accuracy, std_accuracy))

        print(
            f"Batch size: {batch_size}, "
            f"Mean: {mean_accuracy:.4f}, "
            f"Std: {std_accuracy:.4f}"
        )

