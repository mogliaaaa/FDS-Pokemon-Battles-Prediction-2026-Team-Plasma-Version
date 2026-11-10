"""Cross-validation helpers: stratified K-fold with shuffling and fixed seeds."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

SEED = 42
REPEAT_SEEDS = (42, 7, 123)


def cv_scores(model, X, y, seed=SEED, n_splits=5):
    """Accuracy per fold of a (pipeline) model; everything fitted inside the folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_score(clone(model), X, y, cv=cv, scoring="accuracy", n_jobs=-1)


def oof_proba(model, X, y, seed=SEED, n_splits=5):
    """Out-of-fold probability of class 1, on the same folds as ``cv_scores``."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def fold_accuracy(proba, y, seed=SEED, n_splits=5):
    """Accuracy per fold of out-of-fold probabilities, for paired comparisons of blends."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba, y = np.asarray(proba), np.asarray(y)
    return np.array([((proba[val] > 0.5).astype(int) == y[val]).mean() for _, val in cv.split(proba, y)])


def repeated_cv(model, X, y, seeds=REPEAT_SEEDS, n_splits=5):
    """Mean and std of the accuracy over repeated K-fold (one score per fold and seed)."""
    scores = np.concatenate([cv_scores(model, X, y, seed=s, n_splits=n_splits) for s in seeds])
    return scores.mean(), scores.std()


def augmented_cv_scores(model, X, y, extra=(), seed=SEED, n_splits=5, val_transform=None):
    """Accuracy per fold when the training part of each fold gets extra rows of its own battles.

    ``X`` / ``y``: one row per battle (index ``battle_id``), the rows that are validated.
    ``extra``: list of ``(X_extra, weight)``; ``X_extra`` has the columns of ``X`` and is indexed by
    ``battle_id`` (e.g. features of the same battles cut at an earlier turn, experiment 09). Each fold adds
    the extra rows of its training battles only, so a battle is never split between training and
    validation; the folds are the same as ``cv_scores`` with the same seed. ``weight`` goes to the model as
    ``sample_weight``. ``val_transform`` (e.g. ``perturb_p2_ko``) changes the validation part only.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    last_step = model.steps[-1][0]

    def one_fold(train_idx, val_idx):
        ids = X.index[train_idx]
        parts, weights = [X.iloc[train_idx]], [np.ones(len(train_idx))]
        for X_extra, weight in extra:
            rows = X_extra[X_extra.index.isin(ids)]
            parts.append(rows)
            weights.append(np.full(len(rows), weight))
        X_fit = pd.concat(parts)
        m = clone(model).fit(X_fit, y.reindex(X_fit.index), **{f"{last_step}__sample_weight": np.concatenate(weights)})
        X_val = X.iloc[val_idx] if val_transform is None else val_transform(X.iloc[val_idx])
        return (m.predict(X_val) == y.iloc[val_idx].to_numpy()).mean()

    return np.array(Parallel(n_jobs=-1)(delayed(one_fold)(tr, va) for tr, va in cv.split(X, y)))


# --- Robustness: P2's KO recording (EDA, section 5.6) ---------------------------------------

def perturb_p2_ko(X, fraction, seed=SEED):
    """Simulate data where a ``fraction`` of P2's Pokémon vanished with HP <= 0.5 are recorded as KOs.

    Only the KO-count features change; ``p2_lost_est`` (KOs + vanished) is unchanged by
    construction. HP features are left as they are (those Pokémon already had HP <= 0.5), so the
    perturbation is an approximation. Columns missing from ``X`` (e.g. the robust subset) are skipped.
    """
    if "p2_vanished_low50" not in X:
        return X
    X = X.copy()
    rng = np.random.default_rng(seed)
    moved = rng.binomial(X["p2_vanished_low50"].astype(int).to_numpy(), fraction)
    for col, sign in [("p2_ko", 1), ("p2_switch_after_ko", 1), ("p2_vanished", -1), ("p2_vanished_low50", -1)]:
        if col in X:
            X[col] = X[col] + sign * moved
    if "p2_vanished_low25" in X:
        # The lowest-HP ones are assumed to be the KO'd ones
        X["p2_vanished_low25"] = X["p2_vanished_low25"] - np.minimum(moved, X["p2_vanished_low25"])
    if "diff_ko" in X:
        X["diff_ko"] = X["p1_ko"] - X["p2_ko"]
    return X


def stress_test(model, X, y, fractions=(0.5, 1.0), seed=SEED, n_splits=5):
    """Train on clean folds, validate on clean and perturbed folds; returns mean accuracy per setting."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {"clean": []} | {f"p2_ko_fixed_{f:g}": [] for f in fractions}
    for train_idx, val_idx in cv.split(X, y):
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        scores["clean"].append((m.predict(X_val) == y_val).mean())
        for f in fractions:
            scores[f"p2_ko_fixed_{f:g}"].append((m.predict(perturb_p2_ko(X_val, f, seed)) == y_val).mean())
    return {k: float(np.mean(v)) for k, v in scores.items()}
