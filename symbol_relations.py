"""Classify symbol pairs and assemble 388 values for each symbol."""
import keras
import numpy as np
from keras import layers
import tensorflow as tf

from cnn import ROOT, predict_probs

RELATIONS = ("None", "Right", "Sup", "Sub", "Above", "Below", "Inside")


def ordered_geometry(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
        raise ValueError("Supply at least one (x0, y0, x1, y1) box.")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Boxes must be finite and have positive width and height.")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    centers = centers[order]
    sizes = (boxes[:, 2:] - boxes[:, :2])[order]
    h = np.median(sizes[:, 1])
    origin = boxes[:, :2].min(axis=0) - h / 2
    span = boxes[:, 2:].max(axis=0) + h / 2 - origin
    delta = np.diff(centers, axis=0, prepend=centers[:1])
    geo = np.concatenate(((centers-origin)/span, sizes/span,
                          np.clip(delta/h, -4, 4), np.clip(sizes/h, -4, 4)), axis=1)
    return order, geo.astype(np.float32)


@keras.saving.register_keras_serializable(package="physics_ocr")
class RelationHead(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relation = keras.Sequential([layers.Dense(128, activation="relu"), layers.Dropout(.2),
                                          layers.Dense(len(RELATIONS))])
        self.relation_embeddings = self.add_weight(name="relation_embeddings", shape=(7, 16))

    def call(self, inputs, training=False):
        features, probs, geo, mask = inputs
        # Split the first linear layer's input across both symbols by broadcasting.
        n = tf.shape(geo)[1]
        def pairs(x):
            size = tf.stack([tf.shape(x)[0], n, n, tf.shape(x)[-1]])
            return tf.broadcast_to(x[:, :, None, :], size), tf.broadcast_to(x[:, None, :, :], size)
        fa, fb = pairs(features)
        ga, gb = pairs(geo)
        scores = self.relation(tf.concat([fa, fb, ga, gb, gb-ga], -1), training=training)
        valid = mask[:, :, None] & mask[:, None, :] & ~tf.eye(n, dtype=tf.bool)
        embedded = tf.matmul(tf.nn.softmax(scores), self.relation_embeddings)
        embedded *= tf.cast(valid[:, :, :, None], embedded.dtype)
        count = tf.cast(tf.maximum(tf.reduce_sum(tf.cast(mask, tf.int32), 1)-1, 1), embedded.dtype)
        message = tf.concat([tf.reduce_sum(embedded, 1), tf.reduce_sum(embedded, 2)], -1) / count[:, None, None]
        rows = tf.concat([features, probs, geo, message], -1)
        return {"relation_logits": scores, "messages": message, "rows": rows}


def load_relations(path=ROOT / "results/symbol_relations.weights.h5"):
    model = RelationHead()
    inputs = [np.zeros((1,1,d), np.float32) for d in (256,92,8)]
    model(tuple(inputs + [np.ones((1,1), bool)]))
    model.load_weights(path)
    return model


def formula_rows(cnn, relations, crops, boxes):
    order, geo = ordered_geometry(boxes)
    if len(crops) != len(order):
        raise ValueError("Every crop must have one matching box.")
    probs, features = predict_probs(cnn, np.asarray(crops)[order], with_features=True)
    mask = np.ones((1,len(order)), dtype=bool)
    result = relations((features[None], probs[None], geo[None], mask), training=False)
    return {**{key: value.numpy()[0] for key, value in result.items()}, "order": order}
