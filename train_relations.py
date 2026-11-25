"""Train directed relation predictions and 16-dimensional relation embeddings on CROHME."""
import argparse
import json
from pathlib import Path
import h5py
import keras
import numpy as np
import tensorflow as tf

from cnn import ROOT, load_cnn, predict_probs
from symbol_relations import RelationHead, RELATIONS, formula_rows, load_relations, intervening_symbols
from prepare_formula_data import dataset


def batches(rows, rng=None):
    # Bucket by length to reduce the padding needed for pair classification.
    order = np.arange(len(rows))
    if rng is not None:
        rng.shuffle(order)
    chunks = []
    for group in np.array_split(order, max(1,len(order)//160)):
        group = sorted(group, key=lambda i: len(rows[i][3]))
        chunks.extend(group[i:i+8] for i in range(0,len(group),8))
    if rng is not None:
        rng.shuffle(chunks)
    for chunk in chunks:
        n = max(len(rows[i][3]) for i in chunk)
        b = len(chunk)
        arrays = [np.zeros((b,n,d), np.float32) for d in (256,92,8)]
        labels = np.full((b,n), -1, np.int32)
        pairs = np.full((b,n,n), -1, np.int32)
        for j,i in enumerate(chunk):
            f,p,g,y,r = rows[i]
            k = len(y)
            for dst,src in zip(arrays,(f,p,g)):
                dst[j,:k] = src
            labels[j,:k], pairs[j,:k,:k] = y,r
        yield tuple(arrays+[labels>=0]), labels, pairs


def evaluate(model, rows, audit_right=False):
    cm = np.zeros((7,7), np.int64)
    skipped = 0
    correct = total = 0
    for inputs,y,r in batches(rows):
        out = model(inputs, training=False)
        assert out["rows"].shape == (*y.shape,388)
        assert np.isfinite(out["rows"].numpy()).all()
        pred = np.asarray(out["relation_logits"]).argmax(-1)
        mask = r>=0
        if audit_right:
            skipped += int(np.count_nonzero((r==0) & (pred==RELATIONS.index("Right"))
                                            & (intervening_symbols(inputs[2], inputs[3]).numpy()>0)))
        np.add.at(cm,(r[mask],pred[mask]),1)
        correct += int(((inputs[1].argmax(-1)==y) & (y>=0)).sum())
        total += int((y>=0).sum())
    support = cm.sum(1)
    f1 = 2*cm.diagonal()/np.maximum(cm.sum(0)+support,1)
    result = {"relation_macro_f1":float(f1[support>0].mean()), "frozen_cnn_symbol_accuracy":correct/total,
            "relation_accuracy":float(cm.trace()/cm.sum()), "confusion":cm.tolist(),
            "relations":list(RELATIONS), "formulas":len(rows), "symbols":total}
    if audit_right:
        result["false_right_with_intervening_symbol"] = skipped
    return result


def select_negatives(r, limit, miner, inputs):
    candidates = tf.random.shuffle(tf.where(r == 0))
    if miner is None:
        return candidates[:limit]
    # Shuffle first so the random half is uniform among all remaining negatives.
    # Mine only true None pairs. Padding, self-pairs and positives never enter.
    count = tf.minimum(tf.shape(candidates)[0], limit)
    hard_count = count // 2
    logits = tf.gather_nd(miner(inputs, training=False)["relation_logits"], candidates)
    script_probs = tf.gather(tf.nn.softmax(logits), [RELATIONS.index("Sup"), RELATIONS.index("Sub")], axis=-1)
    ranks = tf.argsort(tf.reduce_max(script_probs, axis=-1), direction="DESCENDING", stable=True)
    hard = ranks[:hard_count]
    remaining = tf.scatter_nd(hard[:, None], tf.ones_like(hard), tf.shape(ranks)) == 0
    return tf.concat([tf.gather(candidates, hard), tf.boolean_mask(candidates, remaining)[:count-hard_count]], 0)


ORIGINAL_CNN = ROOT / "results/physics_cnn_92_48_shift_elastic.keras"


def train(epochs, synthetic="none", variant="baseline", cnn_path=ORIGINAL_CNN):
    keras.utils.set_random_seed(42)
    boxes = [] if synthetic != "none" else None
    data = dataset(boxes, cnn_path=cnn_path)
    generator = None
    if boxes is not None:
        from synthetic_relations import SyntheticFormulas
        generator = SyntheticFormulas(data["train"], boxes)
    miner = None
    if variant != "baseline":
        miner = load_relations(ROOT / "results/symbol_relations.weights.h5")
        miner.trainable = False
        keras.utils.set_random_seed(42)
    model = RelationHead(pair_geometry=variant in ("hard_geometry", "between"), count_between=variant == "between")
    optimizer = keras.optimizers.Adam(3e-4, clipnorm=5.)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    sig = [tf.TensorSpec((None,None,d),tf.float32) for d in (256,92,8)]
    mask_sig = tf.TensorSpec((None,None),tf.bool)

    @tf.function(input_signature=[tuple(sig+[mask_sig]),tf.TensorSpec((None,None),tf.int32),
                                  tf.TensorSpec((None,None,None),tf.int32)])
    def step(inputs,y,r):
        positives = tf.where(r>0)
        negatives = select_negatives(r, tf.maximum(16, 2*tf.shape(positives)[0]), miner, inputs)
        with tf.GradientTape() as tape:
            out = model(inputs,training=True)
            chosen = tf.concat([positives,negatives],0)
            pair_loss = loss_fn(tf.gather_nd(r,chosen),tf.gather_nd(out["relation_logits"],chosen))
            supervised = tf.math.divide_no_nan(tf.reduce_sum(pair_loss), tf.cast(tf.size(pair_loss),tf.float32))
            # Decode the learned direction summaries into the annotated relation frequencies.
            # This auxiliary loss trains the embedding even without a downstream sequence model.
            present = tf.cast(r >= 0,tf.float32)
            one_hot = tf.one_hot(r,7)
            incoming, outgoing = tf.split(out["messages"],2,axis=-1)
            message_loss = tf.constant(0.,tf.float32)
            for axis, summary in ((1,incoming),(2,outgoing)):
                counts = tf.reduce_sum(one_hot,axis=axis)
                degree = tf.reduce_sum(present,axis=axis)
                target = tf.math.divide_no_nan(counts,degree[:,:,None])
                logits = tf.matmul(summary,model.relation_embeddings,transpose_b=True)
                ce = tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=logits)
                message_loss += tf.math.divide_no_nan(tf.reduce_sum(ce),tf.reduce_sum(tf.cast(degree>0,tf.float32)))
            loss = supervised + 0.1*message_loss
            tf.debugging.assert_all_finite(loss,"Nonfinite relation loss")
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        return loss

    rng = np.random.default_rng(42)
    history = []
    best = -1
    stem = "symbol_relations" if generator is None else f"symbol_relations_{synthetic}"
    if variant != "baseline":
        stem += f"_{variant}"
    if cnn_path.name != ORIGINAL_CNN.name:
        stem += f"_{cnn_path.stem}"
    path = ROOT/"results"/f"{stem}.weights.h5"
    synthesis_rng = np.random.default_rng(103)
    for epoch in range(epochs):
        rows = data["train"] if generator is None else generator.mix(data["train"],synthetic,synthesis_rng)
        losses = [float(step(x,y,r)) for x,y,r in batches(rows,rng)]
        val = evaluate(model,data["valid"])
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"validation":val})
        print(epoch+1,history[-1]["loss"],val["relation_macro_f1"],val["frozen_cnn_symbol_accuracy"],flush=True)
        score = val["relation_macro_f1"]
        if score > best:
            best = score
            model.save_weights(path)
            if model.pair_geometry:
                with h5py.File(path, "r+") as weights:
                    weights.attrs["pair_geometry"] = True
                    weights.attrs["count_between"] = model.count_between
            best_epoch = epoch+1
    # Use the public reload path for final evaluation as well.
    reloaded = load_relations(path)
    with np.load(ROOT/"data/physics_symbols_92_48.npz") as source:
        images = source["images"][source["valid_indices"][:2]]
    cnn = load_cnn(cnn_path)
    boxes = [[20,0,30,10],[0,0,10,10]]
    example = formula_rows(cnn,reloaded,images,boxes)
    np.testing.assert_array_equal(example["order"], [1,0])
    probs, feats = predict_probs(cnn, images[::-1], with_features=True)
    np.testing.assert_allclose(example["rows"][:,:256], feats, atol=1e-5)
    np.testing.assert_allclose(example["rows"][:,256:348], probs, atol=1e-5)
    assert example["rows"].shape == (2,388) and example["rows"].dtype == np.float32
    single = formula_rows(cnn,reloaded,images[:1],boxes[:1])
    np.testing.assert_array_equal(single["rows"][:,356:], np.zeros((1,32)))
    report = {"best_epoch":best_epoch,"history":history,"test":evaluate(reloaded,data["test"],audit_right=True),
              "training_formulas":len(data["train"]),"cnn":cnn_path.name}
    if variant == "between" and cnn_path.name == ORIGINAL_CNN.name:
        previous = load_relations(ROOT/"results/symbol_relations_hard_geometry.weights.h5")
        report["previous_model_test"] = evaluate(previous,data["test"],audit_right=True)
    report["synthetic_mode"] = synthetic
    report["variant"] = variant
    report["synthetic_formulas_per_epoch"] = len(rows)//4 if generator is not None else 0
    (ROOT/"results"/f"{stem}_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(report["test"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--synthetic",choices=("none","pairs","context"),default="none")
    parser.add_argument("--variant", choices=("baseline", "hard", "hard_geometry", "between"), default="baseline")
    parser.add_argument("--cnn", type=Path, default=ORIGINAL_CNN)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.synthetic != "none" and args.variant != "baseline":
        parser.error("Use real CROHME data for the hard-negative comparison.")
    train(args.epochs,args.synthetic,args.variant,args.cnn)
