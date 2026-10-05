"""Stage 1 - download Fashion-MNIST and save the raw arrays to data/raw/."""
from pathlib import Path

import numpy as np
from tensorflow import keras


def main():
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    np.save(out / "x_train.npy", x_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "x_test.npy", x_test)
    np.save(out / "y_test.npy", y_test)
    print(f"Saved raw data to {out}: train={x_train.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()
