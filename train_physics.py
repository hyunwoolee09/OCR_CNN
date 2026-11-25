"""Prepare MathWriting glyphs and train the shared 48px, 92-class CNN."""

import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import keras
import numpy as np
from PIL import Image
import tensorflow as tf

from cnn import CLASSES, FEATURE_SIZE, IMAGE_SIZE, MODEL_PATH, ROOT, SymbolCNN, crop


BASELINE_ROOT = ROOT
# 기존 32×32 파일에서는 샘플 ID와 라벨 번호만 읽어 기존 분할을 찾아낸다.
# 48×48 픽셀은 압축파일 안의 원본 획에서 다시 만든다.
SOURCE_DATASET = BASELINE_ROOT / "data" / "mathwriting_symbols_32.npz"
ARCHIVE = BASELINE_ROOT / "data" / "mathwriting-2024.tgz"
DATASET = ROOT / "data" / "physics_symbols_92_48.npz"
RESULTS = ROOT / "results"
REPORT_PATH = RESULTS / "physics_results_92_48_keras.json"
SPLITS = ("train", "valid", "test")
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

from prepare_symbols import read_symbol, render_symbol


def split_indices(labels, seed):
    """캐시를 다시 만들 때 기존 클래스별 분할을 그대로 계산한다."""
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLITS}
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        size = len(indices)
        test_size = min(max(1, round(size * 0.2)), size - 2) if size >= 3 else size - 1
        remaining = indices[test_size:]
        valid_size = min(max(1, round(len(remaining) * 0.2)), len(remaining) - 1)
        splits["test"].extend(indices[:test_size])
        splits["valid"].extend(remaining[:valid_size])
        splits["train"].extend(remaining[valid_size:])
    return {name: rng.permutation(indices).tolist() for name, indices in splits.items()}


def prepare_dataset(archive_path=ARCHIVE, output_path=DATASET, cap_scale=False):
    """Render the shared vocabulary from InkML and retain the existing split membership."""
    with np.load(SOURCE_DATASET, allow_pickle=False) as source:
        source_names = source["class_names"].tolist()
        source_labels = source["labels"]
        source_ids = source["sample_ids"]
    # 통합 모델의 기호 순서를 기준으로 새 클래스 번호를 만든다.
    class_ids = {name: index for index, name in enumerate(CLASSES)}
    fraction_label = r"\frac"
    missing = sorted((set(CLASSES) | {fraction_label}) - set(source_names))
    if missing:
        raise ValueError(f"MathWriting is missing required classes: {missing}")

    # 229종 라벨에서 필요한 92종만 골라 새 번호로 바꾼다.
    class_map = np.asarray([class_ids.get(name, -1) for name in source_names])
    mapped_labels = class_map[source_labels]
    keep = mapped_labels >= 0
    original_splits = split_indices(source_labels, SEED)
    # MathWriting의 분수선을 소스 학습 분할에서만 골라 빼기와 같은 클래스로 학습한다.
    fraction_indices = np.asarray(original_splits["train"], dtype=np.int64)
    fraction_indices = fraction_indices[source_labels[fraction_indices] == source_names.index(fraction_label)]
    keep[fraction_indices] = True
    mapped_labels[fraction_indices] = class_ids["-"]
    labels = mapped_labels[keep]
    sample_ids = source_ids[keep]
    image_size = IMAGE_SIZE
    images = np.empty((len(labels), 1, image_size, image_size), dtype=np.float32)
    # 압축파일의 순서와 무관하게 각 기호를 저장할 배열 위치를 찾는다.
    pending = {sample_id: index for index, sample_id in enumerate(sample_ids.tolist())}

    glyph_strokes = [None] * len(images)
    with tarfile.open(archive_path, mode="r|gz", bufsize=1024 * 1024) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.parent.name != "symbols" or path.suffix != ".inkml" or path.stem not in pending:
                continue
            with archive.extractfile(member) as file:
                name, sample_id, strokes = read_symbol(file.read())
            index = pending.pop(path.stem)
            target_name = "-" if name == fraction_label else name
            if sample_id != path.stem or class_ids.get(target_name) != labels[index]:
                raise ValueError(f"Glyph metadata differs from the original dataset: {path.name}")
            glyph_strokes[index] = strokes
            if not pending:
                break
    if pending:
        raise ValueError(f"Could not find {len(pending)} glyphs in the MathWriting archive.")

    # 같은 샘플이 이전과 같은 train/valid/test에 속하도록 기존 번호를 변환한다.
    index_map = np.full(len(source_labels), -1, dtype=np.int64)
    index_map[keep] = np.arange(len(labels))
    splits = {}
    for name, indices in original_splits.items():
        mapped = index_map[indices]
        splits[f"{name}_indices"] = mapped[mapped >= 0]

    # A shared lower bound on physical extent limits magnification without using
    # the symbol label. Estimate it only from training glyphs in raw ink units.
    min_extent = 0.0
    if cap_scale:
        extents = [np.ptp(np.concatenate(glyph_strokes[i]), axis=0).max() for i in splits["train_indices"]]
        min_extent = float(np.median(extents))
    for i, strokes in enumerate(glyph_strokes):
        gray = render_symbol(strokes, image_size=96, min_extent=min_extent)[0]
        ink_image = Image.fromarray(np.rint((1 - gray) * 255).astype(np.uint8))
        images[i] = crop(ink_image, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False)
    temporary = Path(output.name)
    try:
        with output:
            np.savez_compressed(
                output,
                images=images,
                labels=labels,
                class_names=np.asarray(CLASSES),
                sample_ids=sample_ids,
                fraction_bar_indices=index_map[fraction_indices], min_extent=min_extent,
                **splits,
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {len(images)} glyphs with shape {images.shape}.", flush=True)


def load_dataset():
    # 구버전 캐시에는 분수선 학습 표본이 없다. 한 번 다시 만들고 이후에는 재사용한다.
    if DATASET.is_file():
        with np.load(DATASET, allow_pickle=False) as source:
            needs_update = "fraction_bar_indices" not in source.files
    else:
        needs_update = True
    if needs_update:
        prepare_dataset()
    with np.load(DATASET, allow_pickle=False) as source:
        images, labels = source["images"], source["labels"]
        expected_shape = (1, IMAGE_SIZE, IMAGE_SIZE)
        if source["class_names"].tolist() != CLASSES or images.shape[1:] != expected_shape:
            raise ValueError(f"Unexpected class order or image shape in {DATASET.name}.")
        splits = {name: source[f"{name}_indices"].tolist() for name in SPLITS}
    return images, labels, splits


class BestValidationWeights(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.score = (-1.0, float("-inf"))
        self.epoch = 0
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        # 검증 정확도 순으로 선택한다. 정확도가 같으면 검증 손실을 비교한다.
        score = (logs["val_accuracy"], -logs["val_loss"])
        if score > self.score:
            self.score = score
            self.epoch = epoch + 1
            self.weights = self.model.get_weights()


def make_dataset(images, labels, indices, *, training=False, augment_batch=None):
    dataset = tf.data.Dataset.from_tensor_slices((images[indices], labels[indices]))
    if training:
        dataset = dataset.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    if training and augment_batch is not None:
        dataset = dataset.map(augment_batch)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.with_options(options).prefetch(1)


def train_model(
    *, augment_batch=None, model_path=RESULTS / "physics_cnn_92_48.keras", report_path=REPORT_PATH,
    augmentation=None, epochs=EPOCHS, rendered_training=None,
):
    keras.utils.set_random_seed(SEED)
    images, labels, splits = load_dataset()
    rendered_count = 0
    if rendered_training is not None:
        rendered_images, rendered_labels = rendered_training
        rendered_count = len(rendered_labels)
        # 새 표본은 배열 끝에 붙이고, 그 인덱스는 학습 분할에만 추가한다.
        splits["train"].extend(range(len(images), len(images) + rendered_count))
        images = np.concatenate((images, rendered_images))
        labels = np.concatenate((labels, rendered_labels))
    datasets = {
        name: make_dataset(images, labels, indices, training=name == "train", augment_batch=augment_batch)
        for name, indices in splits.items()
    }
    model = SymbolCNN()
    # 정답은 기호 번호 하나이며, 손실 함수에는 softmax 전 점수를 넘긴다.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    best = BestValidationWeights()
    history = model.fit(
        datasets["train"], validation_data=datasets["valid"],
        epochs=epochs, callbacks=[best], shuffle=False, verbose=2,
    ).history

    # 검증에서 선택한 가중치로 시험한다. 정확도는 가장 높은 점수의 기호가 정답과 같은 비율이다.
    model.set_weights(best.weights)
    test = model.evaluate(datasets["test"], return_dict=True, verbose=0)
    test["samples"] = len(splits["test"])
    test["correct"] = round(test["accuracy"] * test["samples"])
    test["accuracy"] = test["correct"] / test["samples"]
    report = {
        "seed": SEED,
        "keras_version": keras.__version__,
        "tensorflow_version": tf.__version__,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE],
        "feature_size": FEATURE_SIZE,
        "class_names": CLASSES,
        "augmentation": augmentation,
        "rendered_training_samples": rendered_count,
        "splits": {name: len(indices) for name, indices in splits.items()},
        "best_epoch": best.epoch,
        "history": history,
        "test": test,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # 모델 구조, 선택된 가중치와 클래스 순서를 .keras 파일에 저장한다.
    model.save(model_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Best epoch: {best.epoch}; test: {test['correct']}/{test['samples']} ({test['accuracy']:.2%})", flush=True)


if __name__ == "__main__":
    train_model()
