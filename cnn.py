"""통합 OCR과 같은 입출력 규격의 Keras 기호 분류기."""

from pathlib import Path

import keras
from keras import layers
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
from symbols import CLASSES


IMAGE_SIZE = 48
FEATURE_SIZE = 256
DROPOUT = 0.2
MODEL_PATH = ROOT / "results" / "physics_cnn_92_48_shift_elastic.keras"


def crop(image, size=IMAGE_SIZE):
    """통합 OCR과 같은 여백, 명암 범위로 기호 이미지를 변환한다."""
    image = image.convert("L")
    bounds = Image.fromarray((np.asarray(image) < 180).astype(np.uint8) * 255).getbbox()
    if bounds:
        image = image.crop(bounds)
    image.thumbnail((size - 6, size - 6), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return (255 - np.asarray(canvas, dtype=np.float32))[None] / 255


@keras.saving.register_keras_serializable(package="physics_ocr")
class SymbolCNN(keras.Model):
    def __init__(
        self, image_size=IMAGE_SIZE, feature_size=FEATURE_SIZE,
        dropout=DROPOUT, class_names=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.feature_size = feature_size
        self.dropout = dropout
        self.class_names = list(CLASSES if class_names is None else class_names)
        self.input_spec = layers.InputSpec(shape=(None, 1, image_size, image_size))
        # 외부 입력은 (N, 1, 48, 48)이다. 합성곱에 들어갈 때 채널을 마지막으로 옮긴다.
        inputs = keras.Input(shape=(1, image_size, image_size))
        x = layers.Permute((2, 3, 1))(inputs)
        for channels in (32, 64, 128):
            for _ in range(2):
                x = layers.Conv2D(channels, 3, padding="same", data_format="channels_last")(x)
                x = layers.GroupNormalization(groups=8, axis=-1, epsilon=1e-5)(x)
                x = layers.ReLU()(x)
            if channels < 128:
                x = layers.MaxPooling2D(2, data_format="channels_last")(x)
        # 고정 입력에서는 12×12를 3×3씩 평균 내면 기존과 같은 4×4가 된다.
        pool_size = image_size // 16
        x = layers.AveragePooling2D(pool_size, data_format="channels_last")(x)
        x = layers.Permute((3, 1, 2))(x)
        x = layers.Reshape((-1,))(x)
        x = layers.Dense(feature_size, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        self.features = keras.Model(inputs, x, name="symbol_features")
        self.classifier = layers.Dense(len(self.class_names), name="symbol_logits")

    def build(self, input_shape):
        self.classifier.build((input_shape[0], self.feature_size))
        super().build(input_shape)

    def extract(self, images, training=False):
        """기호마다 모양 특징 256개와 softmax 이전 기호 점수 92개를 반환한다."""
        features = self.features(images, training=training)
        return features, self.classifier(features)

    def call(self, images, training=False):
        return self.extract(images, training=training)[1]

    def get_config(self):
        return {
            **super().get_config(),
            "image_size": self.image_size,
            "feature_size": self.feature_size,
            "dropout": self.dropout,
            "class_names": self.class_names,
        }


def load_cnn(path=MODEL_PATH):
    """Keras 모델을 읽고 통합 OCR과 라벨 순서 및 입출력 크기를 맞춘다."""
    model = keras.saving.load_model(path, compile=False)
    if (
        model.class_names != CLASSES
        or model.image_size != IMAGE_SIZE
        or model.feature_size != FEATURE_SIZE
    ):
        raise ValueError("Saved CNN does not match the integrated OCR interface.")
    return model


def predict_probs(model, images, with_features=False):
    """입력 순서대로 float32 확률 (N, 92), 필요하면 특징 (N, 256)도 반환한다."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected images shaped (N, 1, {IMAGE_SIZE}, {IMAGE_SIZE}).")
    if model.class_names != CLASSES:
        raise ValueError("CNN class order differs from the shared vocabulary.")
    if len(images) == 0:
        probs = np.empty((0, len(CLASSES)), dtype=np.float32)
        return (probs, np.empty((0, FEATURE_SIZE), dtype=np.float32)) if with_features else probs

    # 추론 모드로 호출하면 Dropout이 꺼진다. softmax는 기호별 점수 축에 적용한다.
    features, logits = model.extract(images, training=False)
    probs = np.asarray(keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1)), dtype=np.float32)
    if with_features:
        features = np.asarray(keras.ops.convert_to_numpy(features), dtype=np.float32)
        return probs, features
    return probs
