import numpy as np


def load_images(path):
    with open(path, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    num_images = int.from_bytes(data[4:8], byteorder="big")
    num_rows = int.from_bytes(data[8:12], byteorder="big")
    num_cols = int.from_bytes(data[12:16], byteorder="big")

    pixels = np.frombuffer(data, dtype=np.uint8, offset=16)

    images = pixels.reshape(num_images, num_rows, num_cols)
    images = images.reshape(num_images, num_rows * num_cols)

    images = images.astype(np.float32) / 255.0

    return images

def load_labels(path):
    with open(path, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    num_labels = int.from_bytes(data[4:8], byteorder="big")

    labels = np.frombuffer(data, dtype=np.uint8, offset=8)

    return labels

def load_mnist(data_dir):
    X_train = load_images(f"{data_dir}/train-images.idx3-ubyte")
    y_train = load_labels(f"{data_dir}/train-labels.idx1-ubyte")

    X_test = load_images(f"{data_dir}/t10k-images.idx3-ubyte")
    y_test = load_labels(f"{data_dir}/t10k-labels.idx1-ubyte")

    return X_train, y_train, X_test, y_test

def train_val_split(X, y, val_size=10000, seed=42):
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, y_train, X_val, y_val

def get_minibatches(X, y, batch_size, rng):
    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]

        yield X[batch_indices], y[batch_indices]