"""Feature selection steps that are fitted inside the CV folds (as the first step of a Pipeline).

- ``CorrelationFilter``: groups features whose absolute Spearman correlation is above a threshold
  (complete-linkage clustering, so every pair inside a group is above it) and keeps one per group.
  It does not look at the target.
- ``top_k_selector``: keeps the K features with the highest gain importance of a quick LightGBM fitted
  on the training part of the fold.
- ``with_selection``: puts a selector in front of a model pipeline.

All steps keep pandas column names, so the selected features can be read back after fitting.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from src.validation import SEED


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Keep one feature per group of highly correlated features (no target used).

    The kept feature of a group is its medoid: the one with the highest mean |r| with the others.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # copy=True: with pandas copy-on-write to_numpy() can return a read-only view
        corr = X.corr(method="spearman").abs().fillna(0.0).to_numpy(copy=True)
        np.fill_diagonal(corr, 1.0)
        dist = squareform(1.0 - corr, checks=False)
        labels = fcluster(linkage(dist, method="complete"), t=1.0 - self.threshold, criterion="distance")
        keep = []
        for lab in np.unique(labels):
            members = np.flatnonzero(labels == lab)
            keep.append(members[np.argmax(corr[np.ix_(members, members)].mean(axis=1))])
        self.columns_ = list(X.columns)
        self.groups_ = {X.columns[m]: [X.columns[i] for i in np.flatnonzero(labels == labels[m])] for m in keep}
        self.support_ = np.isin(np.arange(X.shape[1]), keep)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_) if not isinstance(X, pd.DataFrame) else X
        return X.loc[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)[self.support_]


def top_k_selector(k, seed=SEED):
    """SelectFromModel keeping the K features with the highest gain of a quick LightGBM."""
    ranker = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, importance_type="gain",
                            random_state=seed, n_jobs=1, verbose=-1)
    return SelectFromModel(ranker, max_features=k, threshold=-np.inf)


def with_selection(selector, model):
    """Pipeline: ``selector`` first, then the steps of the model pipeline (output kept as pandas)."""
    model = clone(model)
    steps = model.steps if isinstance(model, Pipeline) else [("model", model)]
    return Pipeline([("select", clone(selector))] + steps).set_output(transform="pandas")


def selected_columns(pipeline):
    """Names of the features kept by the ``select`` step of a fitted pipeline."""
    return list(pipeline.named_steps["select"].get_feature_names_out())
