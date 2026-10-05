"""Stage 2 - normalize pixels to [0, 1], split off a validation set, save to data/processed/."""
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def normalize(x):
    # Part E edits THIS line on both branches to create the merge conflict.
    return x.astype("float32") / 255.0


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    raw, out = Path("data/raw"), Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    x_train = normalize(np.load(raw / "x_train.npy"))
    y_train = np.load(raw / "y_train.npy")
    x_test = normalize(np.load(raw / "x_test.npy"))
    y_test = np.load(raw / "y_test.npy")

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    for name, arr in {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }.items():
        np.save(out / f"{name}.npy", arr)

    print(f"train={x_tr.shape} val={x_val.shape} test={x_test.shape}")


if __name__ == "__main__":
    main()
