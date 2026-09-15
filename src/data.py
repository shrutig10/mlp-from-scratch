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

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist("data")

    X_train_1, y_train_1, X_val_1, y_val_1 = train_val_split(
        X_train,
        y_train,
        val_size=10000,
        seed=42
    )

    X_train_2, y_train_2, X_val_2, y_val_2 = train_val_split(
        X_train,
        y_train,
        val_size=10000,
        seed=42
    )

    print("Same training split:", np.array_equal(X_train_1, X_train_2))
    print("Same training labels:", np.array_equal(y_train_1, y_train_2))
    print("Same validation split:", np.array_equal(X_val_1, X_val_2))
    print("Same validation labels:", np.array_equal(y_val_1, y_val_2))