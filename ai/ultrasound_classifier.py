from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
    / "tn5000_classifier_v2_best.keras"
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Selected using validation data only.
DECISION_THRESHOLD = 0.58


class UltrasoundClassifier:
    """Experimental thyroid-ultrasound image classifier."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        """Load the trained V2 classifier."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found:\n{MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(
            MODEL_PATH
        )

    def preprocess_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        """
        Prepare one image exactly as expected by the V2 model.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            )
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return image_array

    def predict(
        self,
        image_path: Path,
    ) -> dict:
        """Run classifier inference on one thyroid ultrasound image."""

        if self.model is None:
            self.load()

        image_array = self.preprocess_image(
            image_path
        )

        malignant_probability = float(
            self.model.predict(
                image_array,
                verbose=0,
            )[0][0]
        )

        if (
            malignant_probability
            >= DECISION_THRESHOLD
        ):
            prediction = "Malignant"
            confidence = malignant_probability

        else:
            prediction = "Benign"
            confidence = (
                1.0
                - malignant_probability
            )

        return {
            "prediction": prediction,
            "malignant_probability": (
                malignant_probability
            ),
            "benign_probability": (
                1.0
                - malignant_probability
            ),
            "confidence": confidence,
            "threshold": DECISION_THRESHOLD,
        }