"""How usual are the moves of each side, given the species (experiment 16), fitted inside the CV folds.

For each side: the mean over its distinct "species/move" pairs of log p(move | species), with counts from the training
battles of both sides. Battles whose team appears only once tend to use unusual moves. No label is used. In ``fit_transform``
each battle is left out of its own counts; validation and test battles use the counts of all training battles.

The keys come from ``src.features.move_key_features``.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

KEY_P1_MOVES, KEY_P2_MOVES = "key_p1_moves", "key_p2_moves"
PAIR_PRIOR, SPECIES_PRIOR = 0.5, 5.0


def _pairs(key):
    return key.split("|") if isinstance(key, str) and key else []


class MoveTypicalityEncoder(TransformerMixin, BaseEstimator):
    """Replace the two move-key columns with ``p1_move_lp``, ``p2_move_lp`` and ``diff_move_lp``."""

    def fit(self, X, y=None):
        self.pair_count_, self.species_count_ = Counter(), Counter()
        for k1, k2 in zip(X[KEY_P1_MOVES], X[KEY_P2_MOVES]):
            for pair in _pairs(k1) + _pairs(k2):
                self.pair_count_[pair] += 1
                self.species_count_[pair.split("/")[0]] += 1
        return self

    def _side_score(self, key, own_pairs, own_species):
        pairs = _pairs(key)
        if not pairs:
            return np.nan
        scores = [np.log((self.pair_count_[p] - own_pairs[p] + PAIR_PRIOR)
                         / (self.species_count_[p.split("/")[0]] - own_species[p.split("/")[0]] + SPECIES_PRIOR))
                  for p in pairs]
        return float(np.mean(scores))

    def _encode(self, X, leave_out):
        rows = []
        for k1, k2 in zip(X[KEY_P1_MOVES], X[KEY_P2_MOVES]):
            own_pairs, own_species = Counter(), Counter()
            if leave_out:
                own_pairs.update(_pairs(k1) + _pairs(k2))
                own_species.update(p.split("/")[0] for p in _pairs(k1) + _pairs(k2))
            rows.append((self._side_score(k1, own_pairs, own_species), self._side_score(k2, own_pairs, own_species)))
        out = pd.DataFrame(rows, columns=["p1_move_lp", "p2_move_lp"], index=X.index)
        out["diff_move_lp"] = out["p1_move_lp"] - out["p2_move_lp"]
        return pd.concat([X.drop(columns=[KEY_P1_MOVES, KEY_P2_MOVES]), out], axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X)._encode(X, leave_out=True)

    def transform(self, X):
        return self._encode(X, leave_out=False)


def with_move_typicality(model):
    """Pipeline: ``MoveTypicalityEncoder`` first, then the steps of ``model`` (a Pipeline or a bare estimator)."""
    model = clone(model)
    steps = model.steps if isinstance(model, Pipeline) else [("model", model)]
    return Pipeline([("moves", MoveTypicalityEncoder())] + steps)
