"""Stage 4 - evaluate on the test set; write metrics.json and confusion_matrix.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    d = Path("data/processed")
    x_test, y_test = np.load(d / "x_test.npy"), np.load(d / "y_test.npy")

    model = keras.models.load_model("models/model.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=120)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": round(float(loss), 4),
                   "test_accuracy": round(float(acc), 4)}, f, indent=2)
    print(f"test_loss={loss:.4f} test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
