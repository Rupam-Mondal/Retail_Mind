from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

_COMPONENTS = {}


def _get_gradcam_components(model):
    model_key = id(model)
    if model_key in _COMPONENTS:
        return _COMPONENTS[model_key]

    backbone = None
    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            continue
        try:
            layer.get_layer("out_relu")
            backbone = layer
            break
        except ValueError:
            continue

    if backbone is None:
        raise ValueError(
            "A MobileNetV2 backbone with the out_relu layer was not found. "
            "Load the transfer-learning model saved by Day 7."
        )

    last_conv_layer = backbone.get_layer("out_relu")
    backbone_index = model.layers.index(backbone)
    preprocessing_layers = model.layers[1:backbone_index]
    convolution_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=last_conv_layer.output
    )
    classification_layers = model.layers[backbone_index + 1:]

    components = (preprocessing_layers, convolution_model, classification_layers)
    _COMPONENTS[model_key] = components
    return components


def explain_product_image(
    image_path,
    model,
    class_names,
    image_size=None,
    alpha=0.4,
    display=True,
    save_path=None
):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Product image was not found: {image_path}")

    if image_size is None:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        image_size = (int(input_shape[1]), int(input_shape[2]))

    image = tf.keras.utils.load_img(
        image_path,
        target_size=image_size,
        color_mode="rgb"
    )
    image_array = tf.keras.utils.img_to_array(image).astype("float32")
    image_batch = tf.expand_dims(image_array, axis=0)

    probabilities = model(image_batch, training=False).numpy()[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_category = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    preprocessing_layers, convolution_model, classification_layers = _get_gradcam_components(model)

    with tf.GradientTape() as tape:
        preprocessed_image = image_batch
        for layer in preprocessing_layers:
            preprocessed_image = layer(preprocessed_image, training=False)
        convolution_outputs = convolution_model(preprocessed_image, training=False)
        tape.watch(convolution_outputs)

        output = convolution_outputs
        for layer in classification_layers:
            output = layer(output, training=False)

        class_score = output[:, predicted_index]

    gradients = tape.gradient(class_score, convolution_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated for this model.")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(convolution_outputs[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    resized_heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        image_size
    ).numpy().squeeze()
    colored_heatmap = plt.get_cmap("jet")(resized_heatmap)[..., :3]
    overlay = np.clip(
        (1.0 - alpha) * (image_array / 255.0) + alpha * colored_heatmap,
        0.0,
        1.0
    )

    if display or save_path is not None:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(image_array.astype("uint8"))
        axes[0].set_title("Original image")
        axes[1].imshow(resized_heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Heatmap overlay")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            f"Prediction: {predicted_category} | Confidence: {confidence:.2%}"
        )
        figure.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_path, dpi=150, bbox_inches="tight")

        if display:
            plt.show()
        else:
            plt.close(figure)

    return {
        "image_path": str(image_path),
        "predicted_category": predicted_category,
        "confidence": confidence,
        "class_index": predicted_index,
        "probabilities": probabilities,
        "heatmap": heatmap,
        "overlay": overlay
    }
