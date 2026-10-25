import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
NUM_CLASSES = 8

def build_model(backbone_name="ResNet50", num_classes=NUM_CLASSES, trainable=False):
    """
    Builds a feature-extractor transfer learning model with a specified pretrained backbone.
    """
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_layer")

    if backbone_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif backbone_name == "EfficientNetV2S":
        base_model = tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", input_tensor=inputs)
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
    elif backbone_name == "MobileNetV3Large":
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=inputs)
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = trainable

    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"{backbone_name}_FishClassifier")
    return model

if __name__ == "__main__":
    test_model = build_model("ResNet50")
    test_model.summary()
    print("\nModel module built and verified successfully!")
