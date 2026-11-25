"""Train relation, projection and BiLSTM weights on annotated CROHME formulas."""
import argparse
import json
import keras
import numpy as np
import tensorflow as tf

from cnn import ROOT
from formula_encoder import FormulaEncoder, RELATIONS
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


def evaluate(model, rows):
    cm = np.zeros((7,7), np.int64)
    correct = total = 0
    for inputs,y,r in batches(rows):
        out = model(inputs, training=False)
        pred = np.asarray(out["relations"]).argmax(-1)
        mask = r>=0
        np.add.at(cm,(r[mask],pred[mask]),1)
        correct += int(((np.asarray(out["context_logits"]).argmax(-1)==y) & (y>=0)).sum())
        total += int((y>=0).sum())
    support = cm.sum(1)
    f1 = 2*cm.diagonal()/np.maximum(cm.sum(0)+support,1)
    return {"relation_macro_f1":float(f1[support>0].mean()), "symbol_accuracy":correct/total,
            "relation_accuracy":float(cm.trace()/cm.sum()), "confusion":cm.tolist(),
            "relations":list(RELATIONS), "formulas":len(rows), "symbols":total}


def train(epochs):
    keras.utils.set_random_seed(42)
    data = dataset()
    model = FormulaEncoder()
    optimizer = keras.optimizers.Adam(3e-4, clipnorm=5.)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    sig = [tf.TensorSpec((None,None,d),tf.float32) for d in (256,92,8)]
    mask_sig = tf.TensorSpec((None,None),tf.bool)

    @tf.function(input_signature=[tuple(sig+[mask_sig]),tf.TensorSpec((None,None),tf.int32),
                                  tf.TensorSpec((None,None,None),tf.int32)])
    def step(inputs,y,r):
        with tf.GradientTape() as tape:
            out = model(inputs,training=True)
            symbols = tf.boolean_mask(loss_fn(tf.maximum(y,0),out["context_logits"]), y>=0)
            positives = tf.where(r>0)
            negatives = tf.random.shuffle(tf.where(r==0))[:tf.maximum(16,2*tf.shape(positives)[0])]
            chosen = tf.concat([positives,negatives],0)
            pair_loss = loss_fn(tf.gather_nd(r,chosen),tf.gather_nd(out["relations"],chosen))
            loss = tf.reduce_mean(symbols)+tf.math.divide_no_nan(tf.reduce_sum(pair_loss),tf.cast(tf.size(pair_loss),tf.float32))
            tf.debugging.assert_all_finite(loss,"Nonfinite encoder loss")
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        return loss

    rng = np.random.default_rng(42)
    history = []
    best = -1
    path = ROOT/"results/formula_encoder.weights.h5"
    for epoch in range(epochs):
        losses = [float(step(x,y,r)) for x,y,r in batches(data["train"],rng)]
        val = evaluate(model,data["valid"])
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"validation":val})
        print(epoch+1,history[-1]["loss"],val["relation_macro_f1"],val["symbol_accuracy"],flush=True)
        score = (val["relation_macro_f1"]+val["symbol_accuracy"])/2
        if score > best:
            best = score
            model.save_weights(path)
            best_epoch = epoch+1
    # Use the public reload path for final evaluation as well.
    from formula_encoder import load_encoder
    reloaded = load_encoder(path)
    report = {"best_epoch":best_epoch,"history":history,"test":evaluate(reloaded,data["test"]),
              "training_formulas":len(data["train"]),"cnn":"physics_cnn_92_48_shift_elastic.keras"}
    (ROOT/"results/formula_encoder_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(report["test"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    train(args.epochs)
