from pathlib import Path

from ai.ultrasound_classifier import (
    UltrasoundClassifier,
)


classifier = UltrasoundClassifier()

test_image_folder = Path(
    "training_data"
    "/ultrasound"
    "/tn5000"
    "/raw"
    "/dataset"
    "/TN5000_forReview"
    "/JPEGImages"
)

image_files = sorted(
    test_image_folder.glob("*.jpg")
)

if not image_files:
    raise FileNotFoundError(
        "No JPG images found for testing."
    )

selected_image = image_files[0]

print(
    f"Testing image: {selected_image}"
)

result = classifier.predict(
    selected_image
)

print()
print("Prediction result:")
print(result)