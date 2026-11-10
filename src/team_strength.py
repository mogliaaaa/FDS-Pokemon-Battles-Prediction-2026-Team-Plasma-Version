"""Team-strength target encoding (experiment 11), fitted inside the CV folds as the first step of a Pipeline.

Teams repeat across battles: the 10,000 training battles use only about 4,200 distinct P1 teams, and P2's teams
come from the same pool. The win rate of a team, estimated on the training battles, carries information that the turn-30 state does not,
in large part through how rare the team is: a P1 team that appears only once wins about 40% of its battles.

- ``te_p1``: smoothed win rate of P1's exact team;
- ``te_p2``: smoothed win rate of all pool teams that contain every P2 species revealed in the 30 turns (P2's full
  team is unknown, so every compatible team counts);
- ``te_n_p1`` / ``te_n_p2``: log of the number of battles behind each estimate; ``diff_te = te_p1 - te_p2``.

The pool is built from the training rows only: every battle adds P1's team with label ``y`` and, when all six of
P2's Pokémon were revealed, P2's team with label ``1 - y`` (``pooled=True``). Smoothing: (wins + K * 0.5) / (n + K).
Teams never seen (e.g. on a private set with other teams) get the prior 0.5 and a count of 0.

Teams never seen: with the defaults, a model trained where most teams are known learns that "prior 0.5, count 0"
means a rare team, and loses accuracy when many teams are new (notebook 21: -0.59 points with 100% unseen teams).
``counts=False`` drops the two count columns and ``train_unseen`` marks a random share of the training rows as never
seen, so the model also learns what to do without the team information (like dropout).

Experiment 13, off by default (the defaults reproduce submissions 22-24 exactly):
- ``pairs=True`` adds ``pr1``, ``pr2``, ``diff_pr``: every pool team adds y - 0.5 (P1) or 0.5 - y (P2 fully revealed) to
  each of its species pairs, and a team scores the mean over its pairs of sum / (n + 20). A new team made of known
  species still gets a value. In ``fit_transform`` each row is encoded from a pool without its own teams, in either
  role, so the pairs of a training row carry what generalises to a new team. ``train_unseen`` does not touch them;
- ``per_side=True`` draws ``train_unseen`` separately for P1 and P2: a new P1 team usually has a known P2 side.

Experiment 14, off by default:
- ``soft_p2=True``: a P2 team revealed in part (at least ``MIN_REVEALED`` species) also enters the pool, its result spread
  over the compatible pool teams in proportion to their battles. Without it those battles teach nothing about P2's teams;
- ``residual=True``: the pool counts results above expectation, ``y - p + 0.5``, where ``p`` is the out-of-fold probability
  of a small logistic regression on the state at turn 30 (``RESIDUAL_COLUMNS``). Part of a team's result is already explained
  by the state, so the rate gets less noisy and less redundant. The logistic regression sees all the training rows, so a
  row's label reaches its own encoding through that model's coefficients: the change is small and notebook 30 bounds it.

Leakage: ``fit_transform`` (used by the Pipeline on the training part of a fold) encodes each training row with
statistics from the other inner folds only, so no row sees its own label; ``transform`` (validation, test) uses
the statistics of all training rows.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.validation import SEED

KEY_P1, KEY_P2 = "key_p1_team", "key_p2_seen"
TEAM_SIZE = 6
OUTPUT = ["te_p1", "te_p2", "te_n_p1", "te_n_p2", "diff_te"]
PAIR_OUTPUT = ["pr1", "pr2", "diff_pr"]
PAIR_SMOOTHING = 20.0
MIN_REVEALED = 3
RESIDUAL_COLUMNS = ["p1_hp_total", "p2_hp_total", "trend_hp_diff_mean", "p1_ko"]


def _split(key):
    return key.split("|") if isinstance(key, str) and key else []


class _Pool:
    """Win counts of the training teams, with a team x species matrix for superset queries."""

    def __init__(self, keys_p1, keys_p2, y, pooled, pairs=False, soft_p2=False):
        wins, games = {}, {}
        self.pair_sum, self.pair_n = {}, {}
        for k1, k2, label in zip(keys_p1, keys_p2, y):
            wins[k1] = wins.get(k1, 0.0) + label
            games[k1] = games.get(k1, 0) + 1
            if pooled and len(_split(k2)) == TEAM_SIZE:
                wins[k2] = wins.get(k2, 0.0) + 1 - label
                games[k2] = games.get(k2, 0) + 1
            if pairs:
                self._add_pairs(k1, label - 0.5)
                if pooled and len(_split(k2)) == TEAM_SIZE:
                    self._add_pairs(k2, 0.5 - label)
        self.teams = list(games)
        self.row = {t: i for i, t in enumerate(self.teams)}
        self.wins = np.array([wins[t] for t in self.teams])
        self.games = np.array([games[t] for t in self.teams], dtype=float)
        species = sorted({s for t in self.teams for s in _split(t)})
        self.col = {s: j for j, s in enumerate(species)}
        self.member = np.zeros((len(self.teams), len(species)), dtype=bool)
        for i, t in enumerate(self.teams):
            self.member[i, [self.col[s] for s in _split(t)]] = True
        if soft_p2 and pooled:
            self._spread_partial(keys_p2, y)

    def _spread_partial(self, keys_p2, y):
        """Add P2 teams revealed in part: each result goes to the compatible teams, in proportion to their battles."""
        results = {}
        for key, label in zip(keys_p2, y):
            if MIN_REVEALED <= len(_split(key)) < TEAM_SIZE:
                total, count = results.get(key, (0.0, 0))
                results[key] = (total + 1 - label, count + 1)
        base = self.games.copy()
        for key, (total, count) in results.items():
            cols = [self.col.get(s) for s in _split(key)]
            if None in cols:
                continue
            weights = base * self.member[:, cols].all(axis=1)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                self.wins += weights * total
                self.games += weights * count

    def _add_pairs(self, key, value):
        for pair in combinations(_split(key), 2):
            self.pair_sum[pair] = self.pair_sum.get(pair, 0.0) + value
            self.pair_n[pair] = self.pair_n.get(pair, 0) + 1

    def pair_score(self, key):
        """Mean over the species pairs of ``key`` of sum / (n + smoothing); 0 with fewer than two species."""
        scores = [self.pair_sum.get(pair, 0.0) / (self.pair_n.get(pair, 0) + PAIR_SMOOTHING)
                  for pair in combinations(_split(key), 2)]
        return float(np.mean(scores)) if scores else 0.0

    def exact(self, key):
        i = self.row.get(key)
        return (0.0, 0.0) if i is None else (self.wins[i], self.games[i])

    def superset(self, key):
        cols = [self.col.get(s) for s in _split(key)]
        if not cols or None in cols:              # nothing revealed, or a species never seen in the pool
            return 0.0, 0.0
        mask = self.member[:, cols].all(axis=1)
        return self.wins[mask].sum(), self.games[mask].sum()


class TeamStrengthEncoder(TransformerMixin, BaseEstimator):
    """Replace the two team-key columns with the team-strength features (see the module docstring)."""

    def __init__(self, smoothing=10.0, pooled=True, n_inner=5, counts=True, train_unseen=0.0, seed=SEED,
                 pairs=False, per_side=False, soft_p2=False, residual=False):
        self.smoothing = smoothing
        self.pooled = pooled
        self.n_inner = n_inner
        self.counts = counts
        self.train_unseen = train_unseen
        self.seed = seed
        self.pairs = pairs
        self.per_side = per_side
        self.soft_p2 = soft_p2
        self.residual = residual

    def _pool(self, X, y, pairs=False, soft_p2=None):
        soft_p2 = self.soft_p2 if soft_p2 is None else soft_p2
        return _Pool(X[KEY_P1].to_numpy(), X[KEY_P2].to_numpy(), np.asarray(y, dtype=float), self.pooled, pairs, soft_p2)

    def _labels(self, X, y):
        """What the pool counts: the result, or with ``residual=True`` the result above what the state predicts."""
        y = np.asarray(y)
        if not self.residual:
            return y
        columns = [c for c in RESIDUAL_COLUMNS if c in X.columns]
        state_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        expected = cross_val_predict(state_model, X[columns], y.astype(int), cv=cv, method="predict_proba")[:, 1]
        return y - expected + 0.5

    def refit_pool(self, X, y):
        """Build the pool used by ``transform`` from these rows (also used by the stress test on unseen teams)."""
        self.pool_ = self._pool(X, self._labels(X, y), pairs=self.pairs)

    def _encode(self, pool, X):
        k = self.smoothing
        cache_p2 = {}
        rows = []
        for k1, k2 in zip(X[KEY_P1], X[KEY_P2]):
            w1, n1 = pool.exact(k1)
            if k2 not in cache_p2:
                cache_p2[k2] = pool.superset(k2)
            w2, n2 = cache_p2[k2]
            rows.append(((w1 + 0.5 * k) / (n1 + k), (w2 + 0.5 * k) / (n2 + k), np.log1p(n1), np.log1p(n2)))
        out = pd.DataFrame(rows, columns=OUTPUT[:4], index=X.index)
        out["diff_te"] = out["te_p1"] - out["te_p2"]
        return out if self.counts else out.drop(columns=["te_n_p1", "te_n_p2"])

    def _encode_pairs(self, pool, X):
        scores = {k: pool.pair_score(k) for k in set(X[KEY_P1]) | set(X[KEY_P2])}
        out = pd.DataFrame({"pr1": X[KEY_P1].map(scores), "pr2": X[KEY_P2].map(scores)})
        out["diff_pr"] = out["pr1"] - out["pr2"]
        return out

    def _cross_fit_pairs(self, X, y):
        """Pair scores of the training rows: each inner fold is encoded from a pool without its teams, in either role."""
        k1, k2 = X[KEY_P1].to_numpy(), X[KEY_P2].to_numpy()
        parts = []
        for fit_idx, enc_idx in GroupKFold(self.n_inner).split(X, groups=k1):
            banned = set(k1[enc_idx]) | {k for k in k2[enc_idx] if len(_split(k)) == TEAM_SIZE}
            keep = [i for i in fit_idx if k1[i] not in banned and k2[i] not in banned]
            parts.append(self._encode_pairs(self._pool(X.iloc[keep], y[keep], pairs=True, soft_p2=False), X.iloc[enc_idx]))
        return pd.concat(parts).loc[X.index]

    def _combine(self, X, *parts):
        return pd.concat([X.drop(columns=[KEY_P1, KEY_P2]), *parts], axis=1)

    def fit(self, X, y):
        self.feature_names_in_ = np.array(X.columns)
        self.refit_pool(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        y = self._labels(X, y)
        enc = []
        for fit_idx, enc_idx in KFold(self.n_inner, shuffle=True, random_state=self.seed).split(X):
            pool = self._pool(X.iloc[fit_idx], y[fit_idx])
            enc.append(self._encode(pool, X.iloc[enc_idx]))
        enc = pd.concat(enc).loc[X.index]
        if self.train_unseen > 0:
            # Same values a team never seen gets in transform: prior 0.5, count 0
            rng = np.random.default_rng(self.seed)
            hide_p1 = rng.random(len(enc)) < self.train_unseen
            hide_p2 = rng.random(len(enc)) < self.train_unseen if self.per_side else hide_p1
            for hide, te, n in ((hide_p1, "te_p1", "te_n_p1"), (hide_p2, "te_p2", "te_n_p2")):
                enc.loc[hide, te] = 0.5
                if self.counts:
                    enc.loc[hide, n] = 0.0
            enc["diff_te"] = enc["te_p1"] - enc["te_p2"]
        if self.pairs:
            return self._combine(X, enc, self._cross_fit_pairs(X, y))
        return self._combine(X, enc)

    def transform(self, X):
        if self.pairs:
            return self._combine(X, self._encode(self.pool_, X), self._encode_pairs(self.pool_, X))
        return self._combine(X, self._encode(self.pool_, X))

    def get_feature_names_out(self, input_features=None):
        kept = [c for c in self.feature_names_in_ if c not in (KEY_P1, KEY_P2)]
        return np.array(kept + self._output())

    def _output(self):
        base = OUTPUT if self.counts else [c for c in OUTPUT if not c.startswith("te_n_")]
        return base + PAIR_OUTPUT if self.pairs else base


def mask_teams(X, fraction, seed=SEED):
    """Stress test: mark a ``fraction`` of the rows as teams never seen (they get the prior 0.5)."""
    X = X.copy()
    rng = np.random.default_rng(seed)
    hide = rng.random(len(X)) < fraction
    X.loc[hide, [KEY_P1, KEY_P2]] = "__unseen__"
    return X


def with_team_strength(model, **encoder_params):
    """Pipeline: ``TeamStrengthEncoder`` first, then the steps of ``model`` (a Pipeline or a bare estimator)."""
    model = clone(model)
    steps = model.steps if isinstance(model, Pipeline) else [("model", model)]
    return Pipeline([("team", TeamStrengthEncoder(**encoder_params))] + steps)


def unseen_cv_scores(model, X, y, side="both", seed=SEED, n_splits=5):
    """Accuracy per fold when the validation teams are unknown to the encoder.

    After fitting, the encoder's pool is rebuilt without the battles that contain a validation team in either role.
    ``side="p1"`` removes the validation P1 teams, ``side="both"`` also the fully revealed validation P2 teams. Species
    stay known, so the pair scores still work, as on a private set with new teams made of known species. The model
    itself still trains on those battles, so this measures the encoder only. ``model`` needs a step called "team".
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def one_fold(train_idx, val_idx):
        X_fit, y_fit, X_val = X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx]
        m = clone(model).fit(X_fit, y_fit)
        teams = set(X_val[KEY_P1])
        if side == "both":
            teams |= {k for k in X_val[KEY_P2] if len(_split(k)) == TEAM_SIZE}
        keep = ~(X_fit[KEY_P1].isin(teams) | X_fit[KEY_P2].isin(teams))
        m.named_steps["team"].refit_pool(X_fit[keep], y_fit[keep])
        return (m.predict(X_val) == y.iloc[val_idx].to_numpy()).mean()

    return np.array(Parallel(n_jobs=-1)(delayed(one_fold)(tr, va) for tr, va in cv.split(X, y)))
