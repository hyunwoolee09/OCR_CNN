"""Synthetic training rows for the frozen-CNN relation classifier."""
import numpy as np
from cnn import CLASSES
from symbol_relations import RELATIONS, ordered_geometry

RIGHT, SUP, SUB = (RELATIONS.index(r) for r in ("Right", "Sup", "Sub"))
PLUS = CLASSES.index("+")


class SyntheticFormulas:
    def __init__(self, rows, boxes):
        if len(rows) != len(boxes):
            raise ValueError("Expected one set of boxes per training formula.")
        self.glyphs = {}
        self.layouts = {k: [] for k in (RIGHT, SUP, SUB)}
        for (features, probs, _, labels, edges), box in zip(rows, boxes):
            for i, label in enumerate(labels):
                self.glyphs.setdefault(int(label), []).append((features[i], probs[i]))
            for a, b in zip(*np.where((edges >= RIGHT) & (edges <= SUB))):
                if not all(CLASSES[labels[i]].isalnum() for i in (a,b)):
                    continue
                # Preserve real relative size, offset and aspect ratio in base-height units.
                pair = box[[a,b]].copy()
                pair -= np.tile(pair[0,:2], 2)
                pair /= pair[0,3]
                if pair[1,0] + pair[1,2] > pair[0,2]:
                    self.layouts[int(edges[a,b])].append(pair)
        if not all(self.layouts.values()) or PLUS not in self.glyphs:
            raise ValueError("Missing required training glyphs or relationship layouts.")
        self.letters = [k for k in self.glyphs if CLASSES[k].isalpha()]
        self.digits = [k for k in self.glyphs if CLASSES[k].isdigit()]

    @staticmethod
    def pick(items, rng):
        return items[rng.integers(len(items))]

    def triplet(self, rng, with_context):
        labels = [self.pick(self.letters,rng), self.pick(self.digits,rng)]
        if with_context:
            labels.extend([PLUS, self.pick(self.letters,rng)])
        samples = [self.pick(self.glyphs[k],rng) for k in labels]
        features, probs = (np.stack(values) for values in zip(*samples))
        for relation in (RIGHT, SUP, SUB):
            boxes = self.pick(self.layouts[relation],rng).copy()
            edges = np.zeros((len(labels),len(labels)),np.int32)
            np.fill_diagonal(edges,-1)
            edges[0,1] = relation
            if with_context:
                # Keep the plus near the script, but attach it to the baseline.
                x = boxes[:,2].max() + rng.uniform(.02,.30)
                boxes = np.vstack([boxes,[x,.25,x+.5,.75],[x+.65,.05,x+1.3,1.]])
                edges[1 if relation == RIGHT else 0,2] = RIGHT
                edges[2,3] = RIGHT
            order, geo = ordered_geometry(boxes)
            yield features[order], probs[order], geo, np.asarray(labels,np.int32)[order], edges[np.ix_(order,order)]

    def mix(self, real, mode, rng):
        count = len(real)//4
        mixed = [real[i] for i in rng.choice(len(real), len(real)-count, replace=False)]
        while len(mixed) < len(real):
            use_context = mode == "context" and rng.random() < .5
            mixed.extend(self.triplet(rng,use_context))
        return mixed[:len(real)]
