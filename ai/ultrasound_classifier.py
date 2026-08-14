from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from ai.gradcam import make_gradcam_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ultrasound" / "tn5000_classifier_best.keras"
IMAGE_SIZE = 224
DECISION_THRESHOLD = 0.58


class UltrasoundClassifier:
    """Experimental TN5000 thyroid ultrasound benign/malignant classifier."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Ultrasound model not found:\n{MODEL_PATH}")
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found:\n{image_path}")
        image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def predict(self, image_path: Path) -> dict:
        if self.model is None:
            self.load()
        array = self.preprocess_image(image_path)
        malignant_probability = float(self.model.predict(array, verbose=0)[0][0])
        benign_probability = 1.0 - malignant_probability
        prediction = (
            "Malignant" if malignant_probability >= DECISION_THRESHOLD else "Benign"
        )
        confidence = (
            malignant_probability if prediction == "Malignant" else benign_probability
        )
        attention_map = make_gradcam_overlay(
            self.model,
            array,
            image_path,
            binary_positive=(prediction == "Malignant"),
        )
        return {
            "prediction": prediction,
            "benign_probability": benign_probability,
            "malignant_probability": malignant_probability,
            "confidence": confidence,
            "threshold": DECISION_THRESHOLD,
            "attention_map": attention_map,
        }
