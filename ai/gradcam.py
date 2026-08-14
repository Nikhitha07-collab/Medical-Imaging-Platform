from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def _last_spatial_layer(model: tf.keras.Model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            return layer
    raise ValueError("No spatial feature layer was found for Grad-CAM.")


def make_gradcam_overlay(
    model: tf.keras.Model,
    input_array: np.ndarray,
    image_path: Path,
    class_index: int | None = None,
    binary_positive: bool = True,
    alpha: float = 0.42,
) -> np.ndarray:
    """Return an RGB Grad-CAM attention overlay for a Keras image classifier."""
    target_layer = _last_spatial_layer(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(tensor, training=False)
        predictions = tf.convert_to_tensor(predictions)
        if predictions.shape[-1] == 1:
            positive_score = predictions[:, 0]
            score = positive_score if binary_positive else (1.0 - positive_score)
        else:
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]).numpy())
            score = predictions[:, class_index]

    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise ValueError("Grad-CAM gradients could not be computed for this model.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights[0], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    if float(maximum.numpy()) > 0:
        heatmap = heatmap / maximum
    heatmap = heatmap.numpy()

    original = Image.open(image_path).convert("RGB")
    width, height = original.size
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    h = np.asarray(heat, dtype=np.float32) / 255.0

    # Lightweight blue -> yellow -> red heatmap without adding matplotlib.
    red = np.clip(2.0 * h, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * np.abs(h - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - h), 0.0, 1.0)
    color_map = np.stack([red, green, blue], axis=-1) * 255.0

    base = np.asarray(original, dtype=np.float32)
    strength = np.expand_dims(h, axis=-1) * float(alpha)
    overlay = base * (1.0 - strength) + color_map * strength
    return np.uint8(np.clip(overlay, 0, 255))
