from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mri" / "brain_mri_classifier_best.keras"
IMAGE_SIZE = 224
CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]


class MRIClassifier:
    """Experimental three-class brain MRI tumor classifier."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MRI model not found:\n{MODEL_PATH}")
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found:\n{image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32)
        return np.expand_dims(array, axis=0)

    def predict(self, image_path: Path) -> dict:
        if self.model is None:
            self.load()
        array = self.preprocess_image(image_path)
        probs = self.model.predict(array, verbose=0)[0].astype(float)
        index = int(np.argmax(probs))
        return {
            "prediction": CLASS_NAMES[index],
            "meningioma_probability": float(probs[0]),
            "glioma_probability": float(probs[1]),
            "pituitary_probability": float(probs[2]),
            "confidence": float(probs[index]),
        }
