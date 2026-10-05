"""Stage 3 - build and train the ANN using hyperparameters from params.yaml."""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    tf.keras.utils.set_random_seed(p["seed"])

    d = Path("data/processed")
    x_train, y_train = np.load(d / "x_train.npy"), np.load(d / "y_train.npy")
    x_val, y_val = np.load(d / "x_val.npy"), np.load(d / "y_val.npy")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        verbose=2,
    )

    out = Path("models")
    out.mkdir(exist_ok=True)
    model.save(out / "model.h5")
    pd.DataFrame(history.history).to_csv(out / "history.csv", index_label="epoch")
    print("Saved models/model.h5 and models/history.csv")


if __name__ == "__main__":
    main()
