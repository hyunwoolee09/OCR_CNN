"""1픽셀 이동과 약한 탄성 변형을 적용해 손글씨 기호 CNN을 학습한다."""

import tensorflow as tf

from train_physics import RESULTS, SEED, train_model
from train_shift_augmented import AUGMENTATION_SEED, OFFSETS, augment_batch


GRID_SIZE = 5
MAX_DISPLACEMENT = 1.5
PROBABILITY = 0.5
ELASTIC_SEED = SEED + 2
ELASTIC_MODEL = RESULTS / "physics_cnn_92_48_shift_elastic.keras"
ELASTIC_REPORT = RESULTS / "physics_results_92_48_shift_elastic_keras.json"


def elastic_warp(images, offsets):
    """픽셀마다 (세로, 가로) 이동량을 받아 보간한 이미지를 만든다."""
    # 최대 이동량보다 넓게 0을 채워 화면 밖을 참조해도 배경이 나오게 한다.
    margin = 2
    pixels = tf.pad(images[:, 0], ((0, 0), (margin, margin), (margin, margin)))
    height, width = tf.unstack(tf.shape(pixels)[-2:] - 2 * margin)
    rows, columns = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij")
    base = tf.cast(tf.stack((rows, columns), axis=-1), tf.float32) + margin
    coords = base + offsets
    lower = tf.cast(tf.floor(coords), tf.int32)
    rows, columns = tf.unstack(lower, axis=-1)
    row_fraction, column_fraction = tf.unstack(coords - tf.floor(coords), axis=-1)

    def sample(y, x):
        return tf.gather_nd(pixels, tf.stack((y, x), axis=-1), batch_dims=1)

    # 소수점 좌표의 값은 주위 네 픽셀을 가중 평균해서 계산한다.
    top = sample(rows, columns) * (1 - column_fraction) + sample(rows, columns + 1) * column_fraction
    bottom = sample(rows + 1, columns) * (1 - column_fraction) + sample(rows + 1, columns + 1) * column_fraction
    return (top * (1 - row_fraction) + bottom * row_fraction)[:, None]


def augment_elastic_batch(images, labels):
    images, labels = augment_batch(images, labels)
    batch_size = tf.shape(images)[0]
    # 기호마다 5×5 격자에서 이동량을 뽑고, 보간해서 획이 부드럽게 휘게 한다.
    grid = tf.random.uniform(
        (batch_size, GRID_SIZE, GRID_SIZE, 2), minval=-1.0, maxval=1.0, seed=ELASTIC_SEED
    )
    offsets = tf.image.resize(grid, tf.shape(images)[-2:], method="bicubic")
    offsets = tf.clip_by_value(offsets, -1, 1) * MAX_DISPLACEMENT
    apply = tf.random.uniform((batch_size, 1, 1, 1), seed=ELASTIC_SEED + 1) < PROBABILITY
    return elastic_warp(images, tf.where(apply, offsets, 0.0)), labels


def main():
    train_model(
        augment_batch=augment_elastic_batch,
        model_path=ELASTIC_MODEL,
        report_path=ELASTIC_REPORT,
        augmentation={
            "type": "random_one_pixel_shift_and_elastic",
            "splits": ["train"],
            "offsets_dy_dx": OFFSETS,
            "shift_seed": AUGMENTATION_SEED,
            "elastic_seed": ELASTIC_SEED,
            "elastic_grid_size": GRID_SIZE,
            "max_displacement_per_axis_px": MAX_DISPLACEMENT,
            "elastic_probability": PROBABILITY,
            "fill": 0,
        },
    )


if __name__ == "__main__":
    main()
