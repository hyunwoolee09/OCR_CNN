"""학습 이미지에 한 픽셀 이동을 적용하는 Keras 학습 실행 파일."""

import argparse

import tensorflow as tf

from train_physics import RESULTS, SEED, train_model


# (세로 이동, 가로 이동). 원본을 유지하는 경우도 똑같은 확률로 뽑는다.
OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
AUGMENTATION_SEED = SEED + 1
SHIFT_MODEL = RESULTS / "physics_cnn_92_48_shift.keras"
SHIFT_REPORT = RESULTS / "physics_results_92_48_shift_keras.json"


def shift_images(images, choices):
    """(N, 1, H, W) 이미지마다 지정된 방향으로 이동시킨다."""
    padded = tf.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)))
    height, width = images.shape[-2:]
    candidates = []
    for dy, dx in OFFSETS:
        # 아래로 이동시키려면 잘라낼 창을 위로 옮긴다. 빈칸에는 배경값 0이 들어간다.
        top, left = 1 - dy, 1 - dx
        candidates.append(padded[:, :, top:top + height, left:left + width])
    return tf.gather(tf.stack(candidates, axis=1), choices, axis=1, batch_dims=1)


def augment_batch(images, labels):
    choices = tf.random.uniform(
        [tf.shape(images)[0]], maxval=len(OFFSETS), dtype=tf.int32, seed=AUGMENTATION_SEED
    )
    return shift_images(images, choices), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", action="store_true", help="학습 자료에 수학 글꼴 이미지를 추가한다.")
    args = parser.parse_args()
    rendered_training = None
    model_path, report_path = SHIFT_MODEL, SHIFT_REPORT
    if args.rendered:
        from prepare_rendered_symbols import load_rendered_symbols

        rendered_training = load_rendered_symbols()
        model_path = RESULTS / "physics_cnn_92_48_shift_rendered.keras"
        report_path = RESULTS / "physics_results_92_48_shift_rendered_keras.json"

    train_model(
        augment_batch=augment_batch,
        model_path=model_path,
        report_path=report_path,
        rendered_training=rendered_training,
        augmentation={
            "type": "random_one_pixel_shift",
            "splits": ["train"],
            "offsets_dy_dx": OFFSETS,
            "fill": 0,
            "seed": AUGMENTATION_SEED,
        },
    )


if __name__ == "__main__":
    main()
