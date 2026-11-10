"""Candidate models allowed by the competition rules and their hyperparameter search spaces.

Only classical machine learning is allowed: linear models, k-NN, discriminant analysis, naive Bayes,
SVMs, decision trees, bagging / random forests, boosting (sklearn, LightGBM, XGBoost, CatBoost),
ensembles and stacking. Neural networks (including ``MLPClassifier``) and pretrained models are
forbidden, so they are not listed here.

Every model is wrapped in a ``Pipeline`` whose last step is called ``model``: imputation and
scaling are fitted inside the CV folds, and search spaces use the ``model__`` prefix.
Models are single-threaded: parallelism is done by the search over folds and candidates.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.validation import SEED


@dataclass
class ModelSpec:
    """A candidate model: pipeline, search space and search settings."""

    name: str
    family: str
    estimator: Pipeline
    space: dict
    search: str = "random"     # "random" (RandomizedSearchCV) or "grid" (GridSearchCV)
    n_iter: int = 30           # candidates for the random search (scaled by n_iter_scale)
    has_proba: bool = True     # usable in soft voting


def _scaled(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", model)])


def _imputed(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def _raw(model):
    # LightGBM, XGBoost, CatBoost and HistGradientBoosting handle NaN natively
    return Pipeline([("model", model)])


def get_model_specs(seed=SEED):
    """All candidate models, keyed by name."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    specs = [
        # --- Linear models and discriminant analysis ---
        ModelSpec("logreg", "linear", _scaled(LogisticRegression(max_iter=5000, random_state=seed)),
                  {"model__C": loguniform(1e-4, 1e2)}, n_iter=25),
        ModelSpec("ridge", "linear", _scaled(RidgeClassifier(random_state=seed)),
                  {"model__alpha": loguniform(1e-2, 1e4)}, n_iter=25, has_proba=False),
        ModelSpec("linear_svm", "svm", _scaled(LinearSVC(max_iter=20000, random_state=seed)),
                  {"model__C": loguniform(1e-5, 1e0)}, n_iter=25, has_proba=False),
        ModelSpec("lda", "discriminant", _scaled(LinearDiscriminantAnalysis(solver="lsqr")),
                  {"model__shrinkage": [None, "auto", 0.001, 0.01, 0.05, 0.1, 0.3, 0.5]}, search="grid"),
        ModelSpec("qda", "discriminant", _scaled(QuadraticDiscriminantAnalysis()),
                  {"model__reg_param": [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]}, search="grid"),
        ModelSpec("gaussian_nb", "naive_bayes", _scaled(GaussianNB()),
                  {"model__var_smoothing": list(np.logspace(-10, 0, 11))}, search="grid"),
        # --- Distance and kernel methods ---
        ModelSpec("knn", "neighbors", _scaled(KNeighborsClassifier()),
                  {"model__n_neighbors": [5, 11, 21, 31, 51, 75, 101, 151, 201, 301],
                   "model__weights": ["uniform", "distance"], "model__p": [1, 2]}, search="grid"),
        ModelSpec("svm_rbf", "svm", _scaled(SVC(kernel="rbf", cache_size=2000, random_state=seed)),
                  {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-5, 1e-1)},
                  n_iter=20, has_proba=False),
        # --- Trees and bagging ---
        ModelSpec("decision_tree", "tree", _imputed(DecisionTreeClassifier(random_state=seed)),
                  {"model__max_depth": randint(2, 16), "model__min_samples_leaf": randint(5, 300),
                   "model__criterion": ["gini", "entropy"], "model__ccp_alpha": loguniform(1e-6, 1e-2)},
                  n_iter=40),
        ModelSpec("random_forest", "bagging", _imputed(RandomForestClassifier(random_state=seed, n_jobs=1)),
                  {"model__n_estimators": randint(200, 900), "model__max_depth": [None, 8, 12, 16, 24],
                   "model__min_samples_leaf": randint(1, 40), "model__max_features": ["sqrt", 0.1, 0.2, 0.3, 0.5]},
                  n_iter=25),
        ModelSpec("extra_trees", "bagging", _imputed(ExtraTreesClassifier(random_state=seed, n_jobs=1)),
                  {"model__n_estimators": randint(200, 900), "model__max_depth": [None, 8, 12, 16, 24],
                   "model__min_samples_leaf": randint(1, 40), "model__max_features": ["sqrt", 0.1, 0.2, 0.3, 0.5]},
                  n_iter=25),
        # --- Boosting ---
        ModelSpec("adaboost", "boosting",
                  _imputed(AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=seed), random_state=seed)),
                  {"model__n_estimators": randint(100, 800), "model__learning_rate": loguniform(1e-2, 1e0),
                   "model__estimator__max_depth": [1, 2, 3]}, n_iter=20),
        ModelSpec("gradient_boosting", "boosting", _imputed(GradientBoostingClassifier(random_state=seed)),
                  {"model__n_estimators": randint(100, 600), "model__learning_rate": loguniform(1e-2, 2e-1),
                   "model__max_depth": [2, 3, 4, 5], "model__subsample": uniform(0.6, 0.4),
                   "model__min_samples_leaf": randint(10, 150), "model__max_features": ["sqrt", 0.3, 0.5, None]},
                  n_iter=15),
        ModelSpec("hist_gb", "boosting", _raw(HistGradientBoostingClassifier(random_state=seed, early_stopping=False)),
                  {"model__learning_rate": loguniform(5e-3, 2e-1), "model__max_iter": randint(100, 1000),
                   "model__max_leaf_nodes": randint(7, 64), "model__min_samples_leaf": randint(10, 250),
                   "model__l2_regularization": loguniform(1e-3, 10), "model__max_features": uniform(0.3, 0.7)},
                  n_iter=40),
        ModelSpec("lightgbm", "boosting",
                  _raw(LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, subsample_freq=1)),
                  {"model__n_estimators": randint(200, 1500), "model__learning_rate": loguniform(5e-3, 1e-1),
                   "model__num_leaves": randint(7, 64), "model__min_child_samples": randint(10, 200),
                   "model__subsample": uniform(0.5, 0.5), "model__colsample_bytree": uniform(0.2, 0.8),
                   "model__reg_alpha": loguniform(1e-3, 10), "model__reg_lambda": loguniform(1e-3, 10)},
                  n_iter=60),
        ModelSpec("xgboost", "boosting",
                  _raw(XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss")),
                  {"model__n_estimators": randint(200, 1500), "model__learning_rate": loguniform(5e-3, 1e-1),
                   "model__max_depth": randint(2, 8), "model__min_child_weight": loguniform(1, 50),
                   "model__subsample": uniform(0.5, 0.5), "model__colsample_bytree": uniform(0.2, 0.8),
                   "model__gamma": loguniform(1e-3, 5), "model__reg_lambda": loguniform(1e-2, 20)},
                  n_iter=60),
        ModelSpec("catboost", "boosting",
                  _raw(CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False)),
                  {"model__iterations": randint(300, 1500), "model__learning_rate": loguniform(1e-2, 1e-1),
                   "model__depth": randint(3, 9), "model__l2_leaf_reg": loguniform(1, 30),
                   "model__random_strength": uniform(0, 2), "model__bagging_temperature": uniform(0, 1)},
                  n_iter=30),
    ]
    return {s.name: s for s in specs}


def tune(spec, X, y, cv, n_iter_scale=1.0, n_jobs=-1, seed=SEED):
    """Run the hyperparameter search of ``spec`` with accuracy; returns the fitted search object.

    ``refit=False``: we only need the scores and the best parameters, the model is refitted later.
    """
    common = dict(scoring="accuracy", cv=cv, n_jobs=n_jobs, refit=False, error_score=np.nan)
    if spec.search == "grid":
        search = GridSearchCV(clone(spec.estimator), spec.space, **common)
    else:
        n_iter = max(1, int(round(spec.n_iter * n_iter_scale)))
        search = RandomizedSearchCV(clone(spec.estimator), spec.space, n_iter=n_iter, random_state=seed, **common)
    search.fit(X, y)
    return search


def best_from_search(search):
    """(best params, mean CV score, std CV score) read from ``cv_results_``."""
    res = search.cv_results_
    i = int(np.nanargmax(res["mean_test_score"]))
    return res["params"][i], float(res["mean_test_score"][i]), float(res["std_test_score"][i])


def with_params(spec, params):
    """A fresh copy of the spec's pipeline with the given parameters."""
    return clone(spec.estimator).set_params(**params)


def params_to_text(params):
    """Readable 'name=value' string without the 'model__' prefix (for summary tables)."""
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    return ", ".join(f"{k.replace('model__', '')}={fmt(v)}" for k, v in sorted(params.items()))


def params_to_json(params):
    """Make numpy values JSON-serialisable."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


# --- Monotone constraints and seed bagging (experiment 10) ----------------------------------
#
# Directions fixed from the game logic, not from the data: +1 = higher value, P1 more likely to win.
# Features missing from the columns (e.g. P2's loss estimate in the robust subset) are simply skipped.

MONOTONE_DIRECTIONS = {
    "diff_hp_total": 1, "p1_hp_total": 1, "p2_hp_total": -1,
    "trend_hp_diff_t25": 1, "trend_hp_diff_mean": 1,
    "p1_ko": -1, "p1_lost_est": -1, "p2_lost_est": 1, "diff_lost_est": -1,
}
SEED_PARAMS = {"lightgbm": "model__random_state", "xgboost": "model__random_state",
               "catboost": "model__random_seed", "gradient_boosting": "model__random_state"}
BAG_SEEDS = (42, 43, 44, 45, 46)


def monotone_params(name, columns, directions=MONOTONE_DIRECTIONS):
    """Pipeline parameters that make model ``name`` monotone in the given features.

    Supported by LightGBM, XGBoost and CatBoost; scikit-learn's GradientBoosting has no such option ({}).
    ``columns`` must be the columns in the order the model receives them.
    """
    vec = [int(directions.get(c, 0)) for c in columns]
    if name == "lightgbm":
        return {"model__monotone_constraints": vec}
    if name == "catboost":
        # String form "(0,1,-1,...)": CatBoost copies list parameters, which breaks sklearn's clone
        return {"model__monotone_constraints": "(" + ",".join(map(str, vec)) + ")"}
    if name == "xgboost":
        return {"model__monotone_constraints": tuple(vec)}
    return {}


class SeedBagging(ClassifierMixin, BaseEstimator):
    """Average of the predicted probabilities of the same model fitted with several random seeds."""

    def __init__(self, estimator, seed_param="model__random_state", seeds=BAG_SEEDS):
        self.estimator = estimator
        self.seed_param = seed_param
        self.seeds = seeds

    def fit(self, X, y):
        self.estimators_ = [clone(self.estimator).set_params(**{self.seed_param: s}).fit(X, y)
                            for s in self.seeds]
        self.classes_ = self.estimators_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ProbaBlend(ClassifierMixin, BaseEstimator):
    """Weighted mean of the probabilities of several models, each with its own columns (experiment 15).

    ``members`` is a list of ``(name, estimator, columns)`` and ``weights`` a dict ``{name: weight}``, normalised at fit time.
    Weights are fixed by hand, never tuned. A battle is predicted as a win for P1 when the mean probability is above 0.5.
    """

    def __init__(self, members, weights):
        self.members = members
        self.weights = weights

    def fit(self, X, y):
        self.fitted_ = {name: (clone(model).fit(X[columns], y), columns) for name, model, columns in self.members}
        self.classes_ = next(iter(self.fitted_.values()))[0].classes_
        return self

    def member_proba(self, name, X):
        model, columns = self.fitted_[name]
        return model.predict_proba(X[columns])[:, 1]

    def predict_proba(self, X):
        total = sum(self.weights[name] for name in self.fitted_)
        p1 = sum(self.weights[name] * self.member_proba(name, X) for name in self.fitted_) / total
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


MIRROR_PAIRS = [("te_p1", "te_p2"), ("te_n_p1", "te_n_p2"), ("pr1", "pr2")]


def mirror_frame(X):
    """The same battles seen from the other side: p1_/p2_ columns and team scores swapped, diff_ columns negated.

    Columns with no counterpart (for example P1's team statistics, or the trend columns) become missing.
    """
    columns = list(X.columns)
    pairs = [(c, "p2_" + c[3:]) for c in columns if c.startswith("p1_") and "p2_" + c[3:] in columns]
    pairs += [p for p in MIRROR_PAIRS if p[0] in columns]
    out = pd.DataFrame(np.nan, index=X.index, columns=columns, dtype=float)
    for a, b in pairs:
        out[a], out[b] = X[b].to_numpy(dtype=float), X[a].to_numpy(dtype=float)
    negated = [c for c in columns if c.startswith("diff_")]
    out[negated] = -X[negated].to_numpy(dtype=float)
    return out


class MirrorEstimator(ClassifierMixin, BaseEstimator):
    """Train on each battle and on its mirror image with the opposite label; predict by averaging the two views (experiment 16).

    Goes after ``TeamStrengthEncoder`` in a pipeline, so that the team scores are computed inside the folds and then mirrored.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y)
        both = pd.concat([X, mirror_frame(X)], ignore_index=True)
        self.estimator_ = clone(self.estimator).fit(both, np.r_[y, 1 - y])
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        p1 = (self.estimator_.predict_proba(X)[:, 1] + 1 - self.estimator_.predict_proba(mirror_frame(X))[:, 1]) / 2
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class MirrorLinearBoost(ClassifierMixin, BaseEstimator):
    """Boosted trees that start from the logit of a linear model, both fitted on the battles and their mirror images (experiment 17).

    The trees correct what the linear model misses. Their start score is out-of-fold: a battle and its mirror image share an inner fold,
    so neither copy sees its own label. Predictions average the two views of a battle. ``booster`` must be a LightGBM classifier, because
    start scores and raw scores are passed through its own interface. Goes after the encoders in a pipeline, like ``MirrorEstimator``.
    """

    def __init__(self, linear, booster, n_inner=5, seed=SEED):
        self.linear = linear
        self.booster = booster
        self.n_inner = n_inner
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        both = pd.concat([X, mirror_frame(X)], ignore_index=True)
        labels = np.r_[y, 1 - y]
        battles = np.r_[np.arange(len(y)), np.arange(len(y))]
        start = np.zeros(len(labels))
        inner = StratifiedGroupKFold(self.n_inner, shuffle=True, random_state=self.seed)
        for fit_idx, score_idx in inner.split(both, labels, battles):
            start[score_idx] = clone(self.linear).fit(both.iloc[fit_idx], labels[fit_idx]).decision_function(both.iloc[score_idx])
        self.linear_ = clone(self.linear).fit(both, labels)
        self.booster_ = clone(self.booster).fit(both, labels, init_score=start)
        self.classes_ = self.booster_.classes_
        return self

    def _view(self, X):
        raw = self.booster_.predict_proba(X, raw_score=True) + self.linear_.decision_function(X)
        return 1 / (1 + np.exp(-raw))

    def predict_proba(self, X):
        p1 = (self._view(X) + 1 - self._view(mirror_frame(X))) / 2
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def seed_bagged(name, estimator, seeds=BAG_SEEDS):
    """``SeedBagging`` of a pipeline of model ``name`` (uses the right seed parameter for each library)."""
    return SeedBagging(estimator, seed_param=SEED_PARAMS[name], seeds=seeds)
