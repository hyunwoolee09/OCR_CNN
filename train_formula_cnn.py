"""Fine-tune the symbol CNN on formula crops and compare MathWriting render scales."""
import json

import keras
import numpy as np
import tensorflow as tf

from cnn import ROOT, CLASSES, load_cnn
from prepare_formula_data import dataset
from train_elastic_augmented import augment_elastic_batch, ELASTIC_MODEL
from train_physics import make_dataset, SEED


def score(model, images, labels):
    parts = []
    for i in range(0, len(labels), 128):
        pixels = images[i:i+128]
        if pixels.dtype == np.uint8:
            pixels = pixels.astype(np.float32) / 255
        parts.append(np.asarray(model(pixels, training=False)).argmax(axis=-1))
    pred = np.concatenate(parts)
    result = {"correct": int(np.sum(pred == labels)), "samples": len(labels),
              "accuracy": float(np.mean(pred == labels)), "dots": {}}
    for symbol in (".", r"\cdot"):
        idx = CLASSES.index(symbol)
        keep = labels == idx
        result["dots"][symbol] = {"correct": int(np.sum(pred[keep] == idx)), "samples": int(keep.sum()),
                                  "predicted": int((pred == idx).sum())}
    return result


def main():
    crops = dataset(crops_only=True)
    formula_counts = {split: len(rows) for split, rows in crops.items()}
    real = {split: tuple(np.concatenate(col) for col in zip(*rows)) for split, rows in crops.items()}
    del crops
    with np.load(ROOT/"data/physics_symbols_92_48.npz") as source:
        original, labels = source["images"], source["labels"]
        ids = source["sample_ids"]
        splits = {s: source[s+"_indices"] for s in real}
    with np.load(ROOT/"data/physics_symbols_92_48_capped.npz") as source:
        np.testing.assert_array_equal(source["sample_ids"], ids)
        np.testing.assert_array_equal(source["labels"], labels)
        for s, indices in splits.items():
            np.testing.assert_array_equal(source[s+"_indices"], indices)
        capped = source["images"]
        extent = float(source["min_extent"])

    def evaluate(model, render, split):
        idx = splits[split]
        return {"mathwriting": score(model, render[idx], labels[idx]), "crohme": score(model, *real[split])}

    def validation_mean(metrics):
        return (metrics["mathwriting"]["accuracy"] + metrics["crohme"]["accuracy"]) / 2

    original_model = load_cnn(ELASTIC_MODEL)
    baseline = {"validation": evaluate(original_model, original, "valid"), "test": evaluate(original_model, original, "test")}
    report = {"baseline": baseline, "crohme_formulas": formula_counts, "crohme_symbols": {s:len(y) for s,(_,y) in real.items()},
              "min_extent": extent, "epochs": 8, "seed": SEED, "learning_rate": 0.00015}
    scores = {"baseline": validation_mean(baseline["validation"])}
    for variant, render in (("formula_mix", original), ("formula_capped", capped)):
        keras.utils.set_random_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = load_cnn(ELASTIC_MODEL)
        model.compile(optimizer=keras.optimizers.Adam(report["learning_rate"]),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        train_idx = splits["train"]
        x, y = real["train"]
        history = []
        best = -1
        path = ROOT / "results" / (variant+".keras")
        for epoch in range(1, report["epochs"]+1):
            chosen = rng.choice(len(y), len(train_idx), replace=False)
            pixels = np.concatenate([render[train_idx], x[chosen].astype(np.float32)/255])
            targets = np.concatenate([labels[train_idx], y[chosen]])
            train = make_dataset(pixels, targets, np.arange(len(targets)), training=True, augment_batch=augment_elastic_batch)
            model.fit(train, epochs=1, verbose=2)
            validation = evaluate(model, render, "valid")
            history.append(validation)
            value = validation_mean(validation)
            if value > best:
                best, best_epoch = value, epoch
                model.save(path)
        model = load_cnn(path)
        result = {"best_epoch": best_epoch, "history": history, "validation": history[best_epoch-1],
                  "test": evaluate(model, render, "test")}
        if variant == "formula_capped":
            idx = splits["test"]
            result["original_render_test"] = score(model, original[idx], labels[idx])
            result["baseline_on_capped_test"] = score(original_model, capped[idx], labels[idx])
        report[variant] = result
        scores[variant] = best
    report["selected"] = max(scores, key=scores.get)
    report["validation_means"] = scores
    (ROOT/"results/formula_cnn_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print("Selected on validation:", report["selected"], scores, flush=True)


if __name__ == "__main__":
    main()
