from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ct"
    / "covid_ct_classifier_best.keras"
)

IMAGE_SIZE = 224
DECISION_THRESHOLD = 0.46


class CTClassifier:
    """Experimental COVID vs NonCOVID CT image classifier."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "CT classification model not found:\n"
                f"{MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

    def preprocess_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"CT image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        return np.expand_dims(
            array,
            axis=0,
        )

    def predict(
        self,
        image_path: Path,
    ) -> dict:
        if self.model is None:
            self.load()

        array = self.preprocess_image(
            image_path
        )

        raw_prediction = self.model.predict(
            array,
            verbose=0,
        )

        values = np.asarray(
            raw_prediction,
            dtype=np.float32,
        ).reshape(-1)

        if values.size == 1:
            covid_probability = float(values[0])

        elif values.size >= 2:
            # Handles a two-output softmax classifier if the saved model
            # uses [NonCOVID, COVID].
            covid_probability = float(values[-1])

        else:
            raise RuntimeError(
                "CT classifier returned an empty prediction."
            )

        covid_probability = float(
            np.clip(
                covid_probability,
                0.0,
                1.0,
            )
        )

        noncovid_probability = (
            1.0 - covid_probability
        )

        prediction = (
            "COVID"
            if covid_probability >= DECISION_THRESHOLD
            else "NonCOVID"
        )

        confidence = (
            covid_probability
            if prediction == "COVID"
            else noncovid_probability
        )

        return {
            "prediction": prediction,
            "covid_probability": covid_probability,
            "noncovid_probability": noncovid_probability,
            "confidence": confidence,
            "threshold": DECISION_THRESHOLD,
        }
