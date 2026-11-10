"""Empirical move damage estimated from observed HP drops.

Some damaging moves have ``base_power = 0`` in the data (Hidden Power, Return, Frustration,
Seismic Toss, Counter, ...), because their power is variable. Instead of treating them as
harmless, we estimate the expected damage of every move from the HP the defender loses on the
turns the move is used.

The estimate uses only battle states (never ``player_won``), so it does not leak the target and
can be fitted once on the whole training set. It must never be fitted on the test set.

Typical usage::

    from src.data import find_data_dir, load_jsonl
    from src.move_damage import fit_move_damage, annotate_moves

    train = load_jsonl(find_data_dir() / "train.jsonl")
    table = fit_move_damage(train)
    annotate_moves(train, table)   # adds "expected_damage" to every move_details
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Moves that restore the user's HP: the defender's HP change on that turn is not pure damage
HEALING_MOVES = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"}
# Effects that stop (or hide) the damage taken by the defender
BLOCKING_EFFECTS = {"substitute", "protect", "endure"}
SIDES = (("p1", "p2"), ("p2", "p1"))


def collect_damage_observations(battles):
    """Return one row per cleanly attributable move use: move, category, base_power, hp drop.

    A move use is kept only when the drop can be attributed to it: the same defender is active on
    the previous and current turn, it was not already fainted, it did not heal and it was not
    behind Substitute / Protect / Endure. States are recorded at the end of each turn.
    """
    rows = []
    for battle in battles:
        timeline = battle["battle_timeline"]
        for prev, cur in zip(timeline, timeline[1:]):
            for atk, dfn in SIDES:
                move = cur[f"{atk}_move_details"]
                d_prev, d_cur = prev[f"{dfn}_pokemon_state"], cur[f"{dfn}_pokemon_state"]
                d_move = cur[f"{dfn}_move_details"]
                if move is None or d_prev["name"] != d_cur["name"] or d_prev["status"] == "fnt":
                    continue
                if d_move is not None and d_move["name"] in HEALING_MOVES:
                    continue
                if BLOCKING_EFFECTS.intersection(d_cur["effects"]):
                    continue
                rows.append((move["name"], move["category"], move["base_power"],
                             d_prev["hp_pct"] - d_cur["hp_pct"]))
    return pd.DataFrame(rows, columns=["move", "category", "base_power", "hp_drop"])


@dataclass
class MoveDamageTable:
    """Expected damage (fraction of the defender's HP) per move, with fallbacks."""

    table: pd.DataFrame        # index = move; columns n, base_power, mean, median, estimate
    drift: float               # mean HP drop on turns where the attacker used a status move
    bp_intercept: float        # damage ~ bp_intercept + bp_slope * base_power (known-power moves)
    bp_slope: float
    variable_default: float    # fallback for rare variable-power moves (base_power = 0)
    min_count: int

    def expected_damage(self, name, category, base_power):
        """Expected damage of a move; 0 for status moves."""
        if category == "status":
            return 0.0
        if name in self.table.index and self.table.at[name, "n"] >= self.min_count:
            return float(self.table.at[name, "estimate"])
        if base_power > 0:
            return float(max(self.bp_intercept + self.bp_slope * base_power, 0.0))
        return self.variable_default


def fit_move_damage(battles, min_count=30):
    """Fit the expected-damage table on (training) battles.

    The residual HP drift (poison, burn, Leech Seed, Leftovers, ...) is measured on turns where the
    attacker used a status move and subtracted from every drop. Moves seen fewer than
    ``min_count`` times fall back to a linear fit on base power, or to the mean damage of the
    variable-power moves when base power is 0.
    """
    obs = collect_damage_observations(battles)
    drift = obs.loc[obs["category"] == "status", "hp_drop"].mean()
    dmg = obs[obs["category"] != "status"].assign(damage=lambda d: d["hp_drop"] - drift)

    table = dmg.groupby("move").agg(n=("damage", "size"), base_power=("base_power", "first"),
                                    mean=("damage", "mean"), median=("damage", "median"))
    # The mean (misses included) is the expected damage; it already accounts for accuracy
    table["estimate"] = table["mean"].clip(lower=0.0)

    reliable = table[table["n"] >= min_count]
    known = reliable[reliable["base_power"] > 0]
    slope, intercept = np.polyfit(known["base_power"], known["mean"], deg=1, w=np.sqrt(known["n"]))
    variable = dmg[dmg["base_power"] == 0]
    variable_default = float(max(variable["damage"].mean(), 0.0)) if len(variable) else 0.0

    return MoveDamageTable(table=table.sort_values("n", ascending=False), drift=float(drift),
                           bp_intercept=float(intercept), bp_slope=float(slope),
                           variable_default=variable_default, min_count=min_count)


def annotate_moves(battles, damage_table):
    """Add ``expected_damage`` to every non-null move_details, in place. Returns the battles."""
    for battle in battles:
        for turn in battle["battle_timeline"]:
            for side in ("p1", "p2"):
                move = turn[f"{side}_move_details"]
                if move is not None:
                    move["expected_damage"] = damage_table.expected_damage(
                        move["name"], move["category"], move["base_power"])
    return battles
