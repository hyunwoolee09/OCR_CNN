"""Prepare CROHME formula crops with aligned symbol and directed relation targets."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

from cnn import CLASSES, ROOT, crop, load_cnn, predict_probs
from crohme_ink import read_ink
from symbol_relations import ordered_geometry, RELATIONS

URL = "https://raw.githubusercontent.com/ThomasLech/CROHME_extractor/ca2d9328b8ad4dabec94587d8f98f94851ff3a49/data/CROHME_full_v2.zip"
DIGEST = "f9fcbaece8410c76c7fe4c47c15702f0d5e7aa572737805b2e2af596e7de7f75"


def rasterize(strokes, glyphs):
    boxes = np.array([g["box"] for g in glyphs])
    heights = boxes[:, 3] - boxes[:, 1]
    positive = heights[heights > 0]
    if not len(positive):
        raise ValueError("No measurable height")
    # Same scale throughout the formula keeps pen dots small.
    scale = 42 / np.median(positive)
    boxes = boxes * scale + np.array([-1,-1,1,1])
    images = []
    for box, glyph in zip(boxes, glyphs):
        size = np.ceil(box[2:] - box[:2] + 4).astype(int)
        if size.max() > 2048:
            raise ValueError("Excessive glyph extent")
        image = Image.new("L", tuple(size * 3), 255)
        draw = ImageDraw.Draw(image)
        for ref in glyph["refs"]:
            points = (strokes[ref] * scale - box[:2] + 2) * 3
            pixels = [tuple(p) for p in points]
            if len(pixels) > 1:
                draw.line(pixels, 0, width=5, joint="curve")
            for x,y in (pixels[0], pixels[-1]):
                draw.ellipse((x-2,y-2,x+2,y+2), fill=0)
        images.append(crop(image.resize(tuple(size), Image.Resampling.LANCZOS)))
    return np.stack(images), boxes


def dataset(training_boxes=None, *, cnn_path=None, crops_only=False):
    path = ROOT / "data" / "CROHME_full_v2.zip"
    if not path.exists():
        temp = path.with_suffix(".part")
        urlretrieve(URL, temp)
        if sha256(temp.read_bytes()).hexdigest() != DIGEST:
            raise ValueError("CROHME download checksum mismatch")
        temp.replace(path)
    if sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected CROHME archive")
    cnn = None if crops_only else (load_cnn(cnn_path) if cnn_path else load_cnn())
    splits = {key: [] for key in ("train", "valid", "test")}
    rejected = Counter()
    pending = []
    seen = set()
    def flush():
        if not pending:
            return
        crops = np.concatenate([row[1] for row in pending])
        output = [predict_probs(cnn, crops[i:i+128], True) for i in range(0,len(crops),128)]
        probabilities, features = [np.concatenate(col) for col in zip(*output)]
        i = 0
        for split, imgs, geo, labels, relations in pending:
            j = i + len(imgs)
            splits[split].append((features[i:j], probabilities[i:j], geo, labels, relations))
            i = j
        pending.clear()
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist(), key=lambda n: ("TestINKMLGT/" not in n, n)):
            is_test = "/CROHME2013_data/TestINKMLGT/" in name
            if not name.endswith(".inkml") or not (is_test or "/CROHME2013_data/TrainINKML/" in name):
                continue
            try:
                strokes, glyphs, edges, tex = read_ink(archive.read(name))
                if not 1 <= len(glyphs) <= 80:
                    raise ValueError("Formula length")
                labels = [CLASSES.index("-" if g["label"] == r"\frac" else g["label"]) for g in glyphs]
                identity = sha256(b"".join(p.tobytes() for p in strokes.values())).digest()
                if identity in seen:
                    continue
                images, boxes = rasterize(strokes, glyphs)
                order, geo = ordered_geometry(boxes)
                y = np.zeros((len(order),len(order)), dtype=np.int32)
                np.fill_diagonal(y,-1)
                for a,b,r in edges:
                    y[a,b] = RELATIONS.index(r)
                fold = int.from_bytes(sha256(tex.encode()).digest()[:4], "big") % 100
                split = "test" if is_test else ("valid" if fold < 15 else "train")
                ordered_labels = np.array(labels, np.int32)[order]
                if crops_only:
                    splits[split].append((np.rint(images[order]*255).astype(np.uint8), ordered_labels))
                else:
                    pending.append((split, images[order], geo, ordered_labels, y[np.ix_(order,order)]))
                if split == "train" and training_boxes is not None:
                    training_boxes.append(boxes[order])
                seen.add(identity)
            except (ValueError, KeyError, IndexError, ET.ParseError, TypeError):
                rejected["unsupported_or_invalid"] += 1
                continue
            if len(pending) >= 32:
                flush()
        flush()
    if any(not rows for rows in splits.values()):
        raise RuntimeError("Missing CROHME split")
    print("Prepared formulas", {k:len(v) for k,v in splits.items()}, dict(rejected), flush=True)
    return splits
