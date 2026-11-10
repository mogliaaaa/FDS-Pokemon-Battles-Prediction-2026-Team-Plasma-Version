"""Structural preprocessing: normalised tables built from the raw battle JSON.

Nothing here uses ``player_won`` except to carry it along as a column, so every table can be
built for train and test in the same way. The only fitted object, the species table, must be
built from the training set only.

Tables:
- ``team_frame``: one row per P1 team member (plus P2's lead), static information;
- ``timeline_frame``: one row per (battle, turn, side), with the action behind null moves;
- ``pokemon_states``: one row per (battle, side, Pokémon seen), its state after the 30 turns;
- ``build_species_table`` / ``lookup_species``: base stats and types per species, with fallback.
"""
import numpy as np
import pandas as pd

NO_TYPE, NO_STATUS, NO_EFFECT = "notype", "nostatus", "noeffect"
STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ["atk", "def", "spa", "spd", "spe"]
BLOCKING_STATUSES = ["slp", "frz", "par"]
N_TURNS = 30


# --- Normalisation of the data conventions -------------------------------------------------

def normalize_types(types):
    """Drop the 'notype' placeholder and sort, so ['notype', 'psychic'] == ['psychic', 'notype']."""
    return tuple(sorted(t for t in types if t != NO_TYPE))


def normalize_status(status):
    """'nostatus' -> None."""
    return None if status == NO_STATUS else status


def normalize_effect(effects):
    """The effects list always holds exactly one element; 'noeffect' -> None."""
    effect = effects[0] if effects else NO_EFFECT
    return None if effect == NO_EFFECT else effect


# --- Static information ---------------------------------------------------------------------

def team_frame(battles):
    """One row per P1 team member and one per P2 lead (``side`` column), with normalised types."""
    rows = []
    for b in battles:
        members = [("p1", slot, p) for slot, p in enumerate(b["p1_team_details"])]
        members.append(("p2", 0, b["p2_lead_details"]))
        for side, slot, p in members:
            rows.append({"battle_id": b["battle_id"], "side": side, "slot": slot, "name": p["name"],
                         "level": p["level"], "types": normalize_types(p["types"]),
                         "item": p.get("item"), **{s: p[s] for s in STATS},
                         "player_won": b.get("player_won")})
    df = pd.DataFrame(rows)
    df["underlevel"] = df["level"] < 100
    return df


# --- Timeline -------------------------------------------------------------------------------

def timeline_frame(battles, damage_table=None):
    """One row per (battle, turn, side) with normalised state, move and action label.

    If ``damage_table`` (see ``src.move_damage``) is given, an ``expected_damage`` column is added.
    """
    cols = {k: [] for k in ["battle_id", "turn", "side", "name", "hp_pct", "status", "effect",
                            *[f"boost_{b}" for b in BOOSTS], "move", "move_type", "move_category",
                            "move_bp", "move_accuracy", "move_priority", "player_won"]}
    for b in battles:
        bid, won = b["battle_id"], b.get("player_won")
        for t in b["battle_timeline"]:
            for side in ("p1", "p2"):
                st, mv = t[f"{side}_pokemon_state"], t[f"{side}_move_details"]
                cols["battle_id"].append(bid); cols["turn"].append(t["turn"]); cols["side"].append(side)
                cols["name"].append(st["name"]); cols["hp_pct"].append(st["hp_pct"])
                cols["status"].append(normalize_status(st["status"]))
                cols["effect"].append(normalize_effect(st["effects"]))
                for k in BOOSTS:
                    cols[f"boost_{k}"].append(st["boosts"][k])
                cols["move"].append(None if mv is None else mv["name"])
                cols["move_type"].append(None if mv is None else mv["type"])
                cols["move_category"].append(None if mv is None else mv["category"])
                cols["move_bp"].append(np.nan if mv is None else mv["base_power"])
                cols["move_accuracy"].append(np.nan if mv is None else mv["accuracy"])
                cols["move_priority"].append(np.nan if mv is None else mv["priority"])
                cols["player_won"].append(won)
    tl = pd.DataFrame(cols).sort_values(["battle_id", "side", "turn"]).reset_index(drop=True)

    if damage_table is not None:
        # Compute once per distinct move, then map (much faster than per row)
        moves = tl.dropna(subset=["move"]).drop_duplicates(["move", "move_category", "move_bp"])
        lookup = {(m, c, bp): damage_table.expected_damage(m, c, bp)
                  for m, c, bp in moves[["move", "move_category", "move_bp"]].itertuples(index=False)}
        keys = zip(tl["move"], tl["move_category"], tl["move_bp"])
        tl["expected_damage"] = [lookup.get(k, np.nan) for k in keys]
    return label_actions(tl)


def label_actions(tl):
    """Add ``switched``, ``prev_status`` and ``action`` (what happened on the turn) to the timeline.

    action: 'move' | 'switch after KO' | 'voluntary switch' | 'blocked (slp/frz/par)' | 'other'.
    Rules are applied from lowest to highest priority, so later assignments win.
    """
    g = tl.groupby(["battle_id", "side"])
    prev_name = g["name"].shift()
    tl["prev_status"] = g["status"].shift()
    tl["switched"] = prev_name.notna() & (tl["name"] != prev_name)

    action = pd.Series("other", index=tl.index, dtype=object)
    blocked = tl["status"].isin(BLOCKING_STATUSES)
    action[blocked] = "blocked (" + tl.loc[blocked, "status"] + ")"
    action[tl["switched"]] = "voluntary switch"
    action[tl["switched"] & (tl["prev_status"] == "fnt")] = "switch after KO"
    action[tl["move"].notna()] = "move"
    tl["action"] = action
    return tl


def truncate_timeline(tl, n_turns):
    """Keep only the first ``n_turns`` turns of a labelled timeline (experiment 09, snapshots).

    The action labels only look at the previous turn, so they stay the same as if the battle had
    been cut before ``timeline_frame``.
    """
    return tl[tl["turn"] <= n_turns].reset_index(drop=True)


def pokemon_states(tl, n_turns=N_TURNS):
    """Replay the timeline: one row per (battle, side, Pokémon seen) with its final known state.

    ``vanished`` = last seen before the final turn and never marked fainted (left the field and
    did not come back; for P2 often a KO that was not recorded, see the EDA, section 5.6).
    ``n_turns`` is the last turn of the timeline (30, or less for a truncated one).
    """
    keys = ["battle_id", "side", "name"]
    states = tl.groupby(keys).agg(first_turn=("turn", "min"), last_turn=("turn", "max"),
                                  turns_active=("turn", "size"), min_hp=("hp_pct", "min"),
                                  player_won=("player_won", "first")).reset_index()
    # Final state from the last row of each Pokémon (groupby "last" would skip None statuses)
    last_rows = (tl.sort_values("turn", kind="stable").drop_duplicates(keys, keep="last")
                   [keys + ["hp_pct", "status"]]
                   .rename(columns={"hp_pct": "last_hp", "status": "last_status"}))
    states = states.merge(last_rows, on=keys, how="left", validate="one_to_one")
    states["fainted"] = states["last_status"] == "fnt"
    states["vanished"] = (states["last_turn"] < n_turns) & ~states["fainted"]
    return states


# --- Species table (fit on train only) ------------------------------------------------------

def build_species_table(train_battles):
    """Base stats and types per species, from P1 teams and P2 leads of the TRAINING set only."""
    team = team_frame(train_battles)
    table = team.groupby("name").agg(types=("types", "first"), n_seen=("battle_id", "size"),
                                     **{s: (s, "first") for s in STATS})
    return table


def lookup_species(species_table, names):
    """Stats and types for ``names``; unknown species get the mean stats and no types.

    Returns a frame indexed like ``names`` with a ``known_species`` flag.
    """
    out = species_table.reindex(pd.Index(names, name="name"))
    known = out["n_seen"].notna()
    means = species_table[STATS].mean()
    for s in STATS:
        out[s] = out[s].fillna(means[s])
    out["types"] = [t if isinstance(t, tuple) else () for t in out["types"]]
    out["n_seen"] = out["n_seen"].fillna(0).astype(int)
    out["known_species"] = known.to_numpy()
    return out
