# PokePredict

## Pokémon Battle Prediction: FDS Kaggle Competition

Our solution to a supervised **binary classification** challenge: predict whether Player 1 wins a Pokémon Gen 2 OU battle (`player_won`) from an incomplete observation of the match.

## The task

- **Input:** Player 1's full team (six Pokémon), Player 2's lead Pokémon, and the **first 30 active turns** of the battle. Battles last 30-80 turns in total, and the total length is not given.
- **Output:** one binary prediction per test battle (`1` = Player 1 wins, `0` = Player 1 loses).
- **Metric:** Accuracy (private leaderboard).
- **Data:** 10,000 labeled training battles (`train.jsonl`) and 5,000 unlabeled test battles. A Gen 2 type chart is supplied.
- **Final evaluation:** the public leaderboard uses about 30% of the test data, the final standings the other 70%.
- The target is the **final** result, not who is ahead at turn 30: underdogs can win.

### Data fields

| Field | Description |
|-------|-------------|
| `battle_id` | Battle identifier (keep it paired with the prediction) |
| `p1_team_details` | All six Pokémon of Player 1: base stats, types, held items |
| `p2_lead_details` | The Pokémon Player 2 starts with |
| `battle_timeline` | First 30 active turns: Pokémon states and move info |
| `player_won` | Target (train only) |

## Rules (summary)

- Only **classical machine learning**. Neural networks are excluded.
- Only the competition data and the supplied Gen 2 type chart. **No** external databases, extra replay datasets, usage statistics, manually collected tables, or externally pretrained models.
- Never use the test set to select features or tune models. Decisions are based on **local validation** (k-fold on `train.jsonl`); the public leaderboard is only a sanity check.
- Max **3 submissions per day**; **select 3 final submissions** on Kaggle before the competition closes.
- Discussing ideas is fine; private code sharing between teams is not.

## Timeline

| Date | Event |
|------|-------|
| 23/10/2026 | Competition starts |
| **05/11/2026, 23:59** | Competition ends (select 3 final submissions!) |
| **12/11/2026** | Deadline to send public Kaggle notebook + report via the course form |

## Deliverables

1. **Final predictions**: 3 selected submissions on Kaggle in `sample_submission.csv` format.
2. **Public Kaggle notebook**: made public after the competition ends, runs end-to-end, generates the 3 final submissions, **clones this repo** (no copy-pasted code), with a short markdown cell before each submission explaining the approach.
3. **Report**: feature engineering & selection, model experimentation & refinement, validation strategy.
4. **Reproducible code**: this repo, well commented, able to reproduce the final submissions.
