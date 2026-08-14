from pathlib import Path
from typing import Any

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ultrasound"
    / "tn5000_yolo11n_best.pt"
)


class UltrasoundYOLODetector:
    """
    TN5000 thyroid lesion detector using a trained YOLO model.

    This class is responsible only for inference.
    Training is performed separately in Google Colab.
    """

    def __init__(self, model_path: str | Path | None = None):

        self.model_path = Path(
            model_path or DEFAULT_MODEL_PATH
        )

        self.model = None

    def load_model(self) -> None:

        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Ultrasound YOLO model was not found:\n"
                f"{self.model_path}\n\n"
                "Copy the trained Colab best.pt file to this location."
            )

        print("Loading ultrasound YOLO detector:")
        print(self.model_path)

        self.model = YOLO(
            str(self.model_path)
        )

        print(
            "Ultrasound YOLO detector loaded successfully."
        )

    def predict(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:

        self.load_model()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Ultrasound image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            verbose=False,
        )

        if not results:
            return self._empty_result()

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return self._empty_result()

        # Use the highest-confidence detected lesion.
        confidences = result.boxes.conf.cpu().numpy()

        best_index = int(
            confidences.argmax()
        )

        best_box = (
            result.boxes.xyxy[best_index]
            .cpu()
            .numpy()
        )

        confidence = float(
            confidences[best_index]
        )

        x1, y1, x2, y2 = [
            float(value)
            for value in best_box
        ]

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        image_height = int(
            result.orig_shape[0]
        )

        image_width = int(
            result.orig_shape[1]
        )

        image_area = float(
            image_width * image_height
        )

        box_area = (
            width * height
        )

        coverage = (
            box_area / image_area
            if image_area > 0
            else 0.0
        )

        return {
            "detected": True,

            "confidence": confidence,

            "confidence_percent": (
                confidence * 100.0
            ),

            "bbox": {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "width": int(round(width)),
                "height": int(round(height)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
            },

            "coverage": coverage,

            "coverage_percent": (
                coverage * 100.0
            ),

            "image_width": image_width,
            "image_height": image_height,

            "number_of_detections": int(
                len(result.boxes)
            ),

            "model_path": str(
                self.model_path
            ),
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:

        return {
            "detected": False,
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "bbox": None,
            "coverage": 0.0,
            "coverage_percent": 0.0,
            "image_width": None,
            "image_height": None,
            "number_of_detections": 0,
        }


def detect_ultrasound_lesion(
    image_path: str | Path,
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:

    detector = UltrasoundYOLODetector()

    return detector.predict(
        image_path=image_path,
        confidence_threshold=confidence_threshold,
    )