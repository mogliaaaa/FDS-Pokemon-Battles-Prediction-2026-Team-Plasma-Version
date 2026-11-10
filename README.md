# PokePredict: Pokémon Battle Prediction

Team Plasma's solution to the FDS 2026 Kaggle competition: predict whether Player 1 wins a Pokémon Gen 2 OU battle
from a partial view of the match.

This repository contains both our **final submissions** and **the work that led to them**:

- **[`final_notebooks/`](final_notebooks/) holds the three submissions we selected on Kaggle. Start here.**
- [`notebooks/`](notebooks/) holds our experiments: EDA, feature engineering, model comparison, and the ideas that did
  not work as well as those that did.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) is the full experiment log, written as we went (in Italian). For every attempt it
  records what we did, how we validated it, the result and the decision we took.

## Final submissions

| Notebook | Model | Features | CV accuracy, seeds 42/7/123 | CV accuracy, later seeds |
|---|---|---|---|---|
| [`28_submission_ts2_main`](final_notebooks/28_submission_ts2_main.ipynb) | Soft voting of XGBoost, LightGBM, CatBoost and Gradient Boosting | 155 battle features + team strength | 0.6874 | 0.6873 (seeds 1/2/3) |
| [`34_submission_sl_lgbm`](final_notebooks/34_submission_sl_lgbm.ipynb) | LightGBM | As above + Spikes state + move typicality | 0.6922 | 0.6897 (seeds 8/9/10) |
| [`37_submission_mirror_boost`](final_notebooks/37_submission_mirror_boost.ipynb) | LightGBM boosted on a logistic regression, trained on mirrored battles, averaged over 5 random states | As above | 0.6970 | 0.6959 (seeds 11/12/13) |

Each value is the mean of 15 folds (3 seeds × stratified 5-fold). All folds resample the same 10,000 training battles,
so the later seeds are a further check rather than independent data.

Each notebook runs end to end from the raw competition files and explains its approach in a markdown cell before the
submission is generated. The saved outputs are from our runs. The same three files are also in `notebooks/`, next to the
experiments that produced them, and the copies are identical.

## The task

- **Input:** Player 1's full team, Player 2's lead Pokémon, and the first 30 turns of the battle. Battles last 30-80
  turns, and the total length is not given.
- **Target:** `player_won`, the final result of the battle. It is not who is ahead at turn 30.
- **Metric:** accuracy.
- **Data:** 10,000 labelled training battles, 5,000 test battles and a Gen 2 type chart.
- **Rules:** classical machine learning only (no neural networks), competition data only, and no use of the test set to
  select features or tune models.

## Repository layout

```
final_notebooks/    The 3 final submissions
notebooks/          Experiment notebooks, numbered in the order we ran them
src/                Shared code imported by the notebooks
  data.py             Data loading
  preprocessing.py    Normalised tables built from the raw battle JSON
  move_damage.py      Expected move damage estimated from observed HP drops
  features.py         Battle features
  team_strength.py    Team-strength target encoder, fitted inside the CV folds
  move_typicality.py  Move typicality encoder, fitted inside the CV folds
  selection.py        Feature selection steps
  models.py           Allowed models, search spaces, ensembles, mirrored and boosted estimators
  validation.py       Cross-validation helpers
data/processed/     Feature group definitions and tuned parameters
EXPERIMENTS.md      Full experiment log (Italian)
```

## How we validated

- Decisions were based only on stratified 5-fold cross-validation on `train.jsonl`, usually repeated over several seeds
  and compared on paired folds. The public leaderboard (about 30% of the test set) was only a sanity check.
- Every step that learns from the data, including target encoders and feature selection, is part of the model pipeline
  and is refitted on each training fold.
- For most experiments from 06 on, we fixed the threshold for adopting an idea before running it. Several results were
  then confirmed on seeds that had not been used to choose them.
- In three cases we made a submission from a configuration that had not passed the rule (experiments 11, 13 and 15).
  Submission 28 is one of them: its team encoder missed one criterion by 0.013 points. Each exception is recorded in the
  log.

## What we tried

Accuracies are cross-validated. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for seeds, fold counts and the full results.

| # | Experiment | Result | Outcome | Notebook |
|---|---|---|---|---|
| 01 | Baseline: 10 features at turn 30, logistic regression | 0.6622 | First submission | Not in repo |
| 02 | Full feature engineering: 155 features in 7 groups | LightGBM 0.6764 | Reference feature set | [02](notebooks/02_preproc.ipynb), [03](notebooks/03_features.ipynb) |
| 03 | Target encoding of species | +0.0002 at best | Rejected | [03](notebooks/03_features.ipynb) |
| 04 | Tuning of 17 allowed models, ensembles, stress test on P2 KOs | Soft voting 0.6800 | Tuned parameters became the basis for later models | [04](notebooks/04_models.ipynb) |
| 05 | Error analysis on out-of-fold predictions | Well calibrated, no useful candidate feature | Errors are concentrated in battles still open at turn 30 | [08](notebooks/08_error_analysis.ipynb) |
| 06 | 44 team matchup features | +0.22 points | Below threshold | [09](notebooks/09_features_v2.ipynb) |
| 07 | Estimating P2's hidden team from species co-occurrence | No gain | Neutral | Not in repo |
| 08 | Feature selection: top-K, correlation filter, groups, PCA | Top 80: +0.05 points | Neutral | [16](notebooks/16_feature_selection.ipynb) |
| 09 | Earlier-turn snapshots as extra rows, decision-quality features | +0.07 / -0.06 points | Rejected | [17](notebooks/17_snapshots_decisions.ipynb) |
| 10 | Monotonic constraints + averaging over 5 seeds | -0.20 points | Rejected | Not in repo |
| 11 | Team strength: target encoding of teams inside the folds | 0.6869 vs 0.6779 (+0.90 points) | First clear gain | [21](notebooks/21_team_strength.ipynb) |
| 12 | Re-tuning with team strength | No candidate passed the rule | Rejected | [25](notebooks/25_tuning_team_strength.ipynb) |
| 13 | Team encoder with species pairs and per-side masking | 0.6888 vs 0.6867 | Missed the rule narrowly, used by team decision | [26](notebooks/26_encoder_unseen_teams.ipynb) |
| 14 | Further encoder refinements | Best candidate +0.10 points | Rejected | [30](notebooks/30_encoder_refinements.ipynb) |
| 15 | Blending models with different encoders | Rule not met | Blend submitted by team decision after a separate confirmation | [31](notebooks/31_blending.ipynb) |
| 16 | Spikes state, mirrored rows, move typicality | Spikes + typicality 0.6922 vs 0.6888 | Adopted, used in submission 34 | [33](notebooks/33_spikes_mirror_moves.ipynb) |
| 17 | LightGBM boosted on a logistic regression, mirrored battles, averaged random states | 0.6970; +0.59 points over submission 34 on new seeds | Adopted, used in submission 37 | [37](notebooks/37_submission_mirror_boost.ipynb) |

We made more Kaggle submissions than the three we selected. Only the three final submission notebooks are in this
repository, which is why the notebook numbers have gaps. The other submissions are described in the log. Experiments
marked "Not in repo" were run directly in those submission notebooks.

## Running the code

```bash
pip install -r requirements.txt
```

The competition data is not included in the repository. Put `train.jsonl`, `test.jsonl`, `type_chart_gen2.csv` and
`sample_submission.csv` in `data/raw/`.

**Final submissions.** Run any notebook in `final_notebooks/`. They need only the raw files and write to
`submissions/`. On Kaggle, attach the competition data and turn on internet access: the notebook clones this repository
and writes `/kaggle/working/submission.csv`.

**Experiments.** Most notebooks in `notebooks/` read intermediate files that are not committed, so run them in this
order: `02_preproc` (tables), then `03_features` (features). `08_error_analysis` writes the out-of-fold predictions that
some later notebooks use for residual checks. `04_models` takes about 1.5 hours and is only needed to regenerate the
tuned parameters, which are already in `data/processed/models/`.

## License

[MIT](LICENSE)
