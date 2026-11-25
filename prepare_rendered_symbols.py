"""LaTeX 표기의 개별 기호를 CNN 학습용 인쇄체 이미지로 만든다."""

import numpy as np
from PIL import Image

from cnn import CLASSES, IMAGE_SIZE, ROOT, crop


FONTSETS = ("cm", "stix", "dejavusans")
RENDERED_DATASET = ROOT / "data" / "rendered_symbols_92_48.npz"


def prepare_rendered_symbols():
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    parser = MathTextParser("agg")
    samples = {}
    ambiguous = set()
    for font in FONTSETS:
        properties = FontProperties(size=30, math_fontfamily=font)
        for label, symbol in enumerate(CLASSES):
            # MathText에서 근호는 인수가 필수다. 공백을 넣어 근호만 그린다.
            expression = r"\sqrt{\,}" if symbol == r"\sqrt" else symbol
            rendered = parser.parse("$" + expression + "$", dpi=150, prop=properties)
            # MathText의 글씨 강도를 흰 배경 이미지로 바꿔 기존 CNN 전처리를 적용한다.
            image = crop(Image.fromarray(255 - np.asarray(rendered.image)))
            if not image.any():
                raise ValueError(f"Empty rendering for {symbol!r} in {font}.")
            key = image.tobytes()
            if key in samples:
                if samples[key][1] != label:
                    ambiguous.add(key)
                continue
            samples[key] = (image, label, font)

    # 같은 픽셀의 반복과, 같은 모양에 정답이 두 개인 합성 표본을 제거한다.
    images, labels, fonts = zip(*(row for key, row in samples.items() if key not in ambiguous))
    RENDERED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RENDERED_DATASET,
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(CLASSES),
        fonts=np.asarray(fonts),
    )
    print(f"Prepared {len(labels)} rendered symbols covering {len(set(labels))} classes.", flush=True)


def load_rendered_symbols():
    if not RENDERED_DATASET.is_file():
        prepare_rendered_symbols()
    with np.load(RENDERED_DATASET, allow_pickle=False) as data:
        images, labels = data["images"], data["labels"]
        if (
            data["class_names"].tolist() != CLASSES
            or images.shape != (len(labels), 1, IMAGE_SIZE, IMAGE_SIZE)
            or labels.shape != (len(images),)
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.all((labels >= 0) & (labels < len(CLASSES)))
        ):
            raise ValueError("Rendered data does not match the CNN input or label order.")
    return images, labels


if __name__ == "__main__":
    prepare_rendered_symbols()
