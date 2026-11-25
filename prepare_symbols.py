"""Render MathWriting symbols once and save arrays for CNN training."""

from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE = DATA_DIR / "mathwriting-2024.tgz"
# 이 파일을 직접 실행하면 전체 229종의 32×32 자료와 확인용 이미지를 만든다.
# 48×48 모델은 이 파일의 함수들을 불러서 필요한 기호의 원본 획을 다시 그린다.
DATASET = DATA_DIR / "mathwriting_symbols_32.npz"
PREVIEW = DATA_DIR / "symbol_examples.png"
IMAGE_SIZE = 32
PADDING = 2
STROKE_WIDTH = 2
SUPERSAMPLE = 4
EXPECTED_SYMBOLS = 6423


def read_symbol(inkml):
    """Return the symbol label, sample ID, and XY coordinates for each stroke."""
    root = ET.fromstring(inkml)
    # InkML의 annotation에는 기호 이름과 샘플 ID가 따로 들어 있다.
    annotations = {item.get("type"): item.text for item in root.findall("{*}annotation")}
    label = annotations["label"].strip()
    sample_id = annotations["sampleId"].strip()
    strokes = []

    for trace in root.findall("{*}trace"):
        # trace 하나가 펜 획 하나다. 각 점에서 x, y, 시간 중 좌표 두 값만 사용한다.
        points = [point.split() for point in (trace.text or "").split(",") if point.strip()]
        if not points:
            continue
        if any(len(point) != 3 for point in points):
            raise ValueError(f"Expected x, y, and time in {sample_id}")
        xy = np.asarray(points, dtype=np.float64)[:, :2]
        if not np.isfinite(xy).all():
            raise ValueError(f"Invalid stroke coordinates in {sample_id}")
        strokes.append(xy)

    if not label or not sample_id or not strokes:
        raise ValueError(f"Missing label, sample ID, or strokes in {sample_id!r}")
    return label, sample_id, strokes


def render_symbol(strokes, image_size=IMAGE_SIZE, min_extent=0.0):
    """Center the glyph and return one channel of grayscale pixels in [0, 1]."""
    # 모든 획의 바깥 상자를 구한다. 전체 기호의 가로세로 비율을 유지한다.
    points = np.concatenate(strokes)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    extent = float((high - low).max())

    # 먼저 큰 캔버스에 그렸다가 줄이면 획 가장자리에 회색 단계가 생긴다.
    side = image_size * SUPERSAMPLE
    scale_factor = image_size / IMAGE_SIZE
    pen_width = round(STROKE_WIDTH * SUPERSAMPLE * scale_factor)
    padding = round(PADDING * scale_factor)
    usable_span = (image_size - 2 * padding) * SUPERSAMPLE - pen_width
    extent = max(extent, min_extent)
    scale = usable_span / extent if extent else 1.0
    canvas = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(canvas)
    radius = (pen_width - 1) / 2

    for stroke in strokes:
        # 같은 확대 비율과 이동량을 모든 획에 적용한다. 점과 본체의 상대 배치도 유지된다.
        pixels = (stroke - center) * scale + (side - 1) / 2
        xy = [tuple(point) for point in pixels]
        if len(xy) > 1:
            draw.line(xy, fill=255, width=pen_width, joint="curve")
        # 획의 끝을 둥글게 그린다. 좌표가 하나뿐인 획도 이 과정에서 점으로 표시된다.
        for x, y in (xy[0], xy[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    image = canvas.resize((image_size, image_size), Image.Resampling.LANCZOS)
    # [None, :, :]로 채널 축을 추가한다. 배경은 0, 진한 획은 1에 가깝다.
    return np.asarray(image, dtype=np.float32)[None, :, :] / 255.0


def load_symbols(archive):
    rows = []
    # 압축 전체를 풀지 않고 symbols/ 안의 InkML 파일만 순서대로 읽는다.
    with tarfile.open(archive, mode="r|gz", bufsize=1024 * 1024) as source:
        for member in source:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.parent.name != "symbols" or path.suffix != ".inkml":
                continue
            with source.extractfile(member) as inkml:
                label, sample_id, strokes = read_symbol(inkml.read())
            if sample_id != path.stem:
                raise ValueError(f"Sample ID does not match file name: {path}")
            rows.append((sample_id, label, render_symbol(strokes)))
            if len(rows) == EXPECTED_SYMBOLS:
                break

    if len(rows) != EXPECTED_SYMBOLS or len({row[0] for row in rows}) != EXPECTED_SYMBOLS:
        raise ValueError(f"Expected {EXPECTED_SYMBOLS} distinct symbols, found {len(rows)} files")

    # 샘플 ID 순서로 배열을 고정하고, 기호 이름을 정수 라벨 번호로 바꾼다.
    rows.sort(key=lambda row: row[0])
    sample_ids = np.asarray([sample_id for sample_id, _, _ in rows])
    class_names, labels = np.unique([label for _, label, _ in rows], return_inverse=True)
    images = np.stack([image for _, _, image in rows])
    return images, labels, class_names, sample_ids


def save_dataset(images, labels, class_names, sample_ids):
    # 임시 파일에 쓰기를 마친 뒤 이름을 바꿔서 완성된 데이터만 남긴다.
    with tempfile.NamedTemporaryFile(dir=DATA_DIR, suffix=".npz", delete=False) as output:
        temporary_path = Path(output.name)
        try:
            np.savez_compressed(
                output, images=images, labels=labels, class_names=class_names, sample_ids=sample_ids
            )
            output.close()
            temporary_path.replace(DATASET)
        finally:
            output.close()
            temporary_path.unlink(missing_ok=True)


def save_preview():
    """Show three saved images enlarged with nearest-neighbor sampling."""
    examples = ("4a9b3ed87c5e10af", "971f42e33222eb43", "931922164c74dfc9")
    with np.load(DATASET) as dataset:
        images = dataset["images"]
        sample_ids = dataset["sample_ids"]
        names = dataset["class_names"][dataset["labels"]]

    tile = IMAGE_SIZE * 5
    gap = 20
    sheet = Image.new("RGB", (3 * tile + 4 * gap, tile + 83), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default(size=23)
    id_font = ImageFont.load_default(size=12)

    for column, sample_id in enumerate(examples):
        index = int(np.searchsorted(sample_ids, sample_id))
        # 사람 눈으로 보기 편하게 미리보기만 흰 배경과 검은 획으로 표시한다.
        gray = np.rint((1.0 - images[index, 0]) * 255).astype(np.uint8)
        image = Image.fromarray(gray).resize((tile, tile), Image.Resampling.NEAREST)
        x = gap + column * (tile + gap)
        y = gap
        sheet.paste(image, (x, y))
        draw.rectangle((x - 1, y - 1, x + tile, y + tile), outline="#cbd5e1")
        draw.text((x + tile / 2, y + tile + 21), str(names[index]), anchor="mm", font=label_font, fill="#172033")
        draw.text((x + tile / 2, y + tile + 42), sample_id, anchor="mm", font=id_font, fill="#475569")

    sheet.save(PREVIEW)


def main():
    images, labels, class_names, sample_ids = load_symbols(ARCHIVE)
    save_dataset(images, labels, class_names, sample_ids)
    save_preview()
    print(f"Saved {len(images):,} images with shape {images.shape} and dtype {images.dtype}")
    print(f"Labels: {len(class_names)} classes")
    print(f"Dataset: {DATASET.name}")
    print(f"Preview: {PREVIEW.name}")


if __name__ == "__main__":
    main()
