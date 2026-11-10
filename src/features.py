"""Feature engineering: one row per battle, built from the preprocessing tables.

Inputs are the tables produced by ``src.preprocessing`` (see ``notebooks/02_preproc.ipynb``):
``team`` (team_frame), ``timeline`` (timeline_frame with expected_damage), ``states``
(pokemon_states), plus the species table (train only) and the type chart.

Features are computed per side as ``p1_*`` / ``p2_*`` and, where it makes sense, as
``diff_* = p1 - p2``. Because P2's KOs are under-recorded (EDA, section 5.6), P1 and P2 columns are
always kept as well, and ``robust_columns`` drops every feature that relies on P2's recorded KOs.

Nothing here is fitted on the target: species target encoding must be done inside the CV folds,
so species names are only returned as categorical columns (group "species_cat").
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.move_damage import HEALING_MOVES
from src.preprocessing import BOOSTS, N_TURNS, STATS, lookup_species, pokemon_states, truncate_timeline

TEAM_SIZE = 6
STATUSES = {"slp": ["slp"], "par": ["par"], "psn": ["psn", "tox"], "brn": ["brn"], "frz": ["frz"]}
EFFECTS = ["substitute", "leechseed", "confusion", "trapped", "perish"]
MOVE_GROUPS = {
    "sleep_moves": {"lovelykiss", "sleeppowder", "hypnosis", "spore", "sing"},
    "para_moves": {"thunderwave", "stunspore", "glare"},
    "toxic_moves": {"toxic"},
    "heal_moves": HEALING_MOVES | {"painsplit"},
    "setup_moves": {"curse", "swordsdance", "growth", "bellydrum", "agility", "acidarmor", "defensecurl"},
    "hazard_moves": {"spikes"},
    "phaze_moves": {"roar", "whirlwind"},
    "explosion_moves": {"explosion", "selfdestruct"},
}
COUNTER_MOVES = {"counter", "mirrorcoat"}
ACTIONS = {"switch after KO": "switch_after_ko", "voluntary switch": "voluntary_switch",
           "blocked (slp)": "blocked_slp", "blocked (par)": "blocked_par", "blocked (frz)": "blocked_frz",
           "other": "other_null"}


def _wide(df, value_cols):
    """(battle_id, side) rows -> one row per battle with p1_* / p2_* columns."""
    w = df.set_index(["battle_id", "side"])[value_cols].unstack("side")
    w.columns = [f"{side}_{col}" for col, side in w.columns]
    return w


def _add_diffs(df, names):
    for n in names:
        df[f"diff_{n}"] = df[f"p1_{n}"] - df[f"p2_{n}"]
    return df


# --- Static information ---------------------------------------------------------------------

def static_features(team):
    """P1 team summary and P2 lead stats (weak on their own, see the EDA)."""
    p1 = team[team["side"] == "p1"].assign(bst=lambda d: d[STATS].sum(axis=1))
    g = p1.groupby("battle_id")
    out = pd.concat([g[STATS + ["bst"]].mean().add_prefix("p1_team_mean_"),
                     g[["base_spe", "bst"]].max().add_prefix("p1_team_max_"),
                     g[["bst"]].min().add_prefix("p1_team_min_")], axis=1)
    out["p1_n_underlevel"] = g["underlevel"].sum()
    out["p1_min_level"] = g["level"].min()
    out["p1_n_leftovers"] = g["item"].apply(lambda s: (s == "leftovers").sum())
    out["p1_n_distinct_types"] = g["types"].apply(lambda s: len({t for ts in s for t in ts}))

    lead = team[team["side"] == "p2"].set_index("battle_id")
    out = out.join(lead[STATS].add_prefix("p2_lead_"))
    out["p2_lead_bst"] = lead[STATS].sum(axis=1)
    out["p2_lead_level"] = lead["level"]
    return out


# --- State after 30 turns -------------------------------------------------------------------

def state_features(states):
    """KOs, revealed / vanished Pokémon, remaining HP and statuses per side after 30 turns."""
    s = states.assign(
        hp_left=np.where(states["fainted"], 0.0, states["last_hp"]),
        statused=states["last_status"].notna() & ~states["fainted"],
        vanished_low25=states["vanished"] & (states["last_hp"] <= 0.25),
        vanished_low50=states["vanished"] & (states["last_hp"] <= 0.5),
    )
    g = s.groupby(["battle_id", "side"])
    per_side = pd.DataFrame({
        "seen": g.size(), "ko": g["fainted"].sum(), "hp_seen": g["hp_left"].sum(),
        "statused": g["statused"].sum(), "vanished": g["vanished"].sum(),
        "vanished_low25": g["vanished_low25"].sum(), "vanished_low50": g["vanished_low50"].sum(),
    }).reset_index()
    per_side["unseen"] = TEAM_SIZE - per_side["seen"]
    # Unseen Pokémon are assumed at full HP
    per_side["hp_total"] = per_side["hp_seen"] + per_side["unseen"]
    # Combined loss estimate: stays meaningful whether KOs are recorded or not (EDA, section 5.6)
    per_side["lost_est"] = per_side["ko"] + per_side["vanished_low50"]
    cols = ["seen", "ko", "hp_total", "statused", "vanished", "vanished_low25",
            "vanished_low50", "lost_est"]
    out = _wide(per_side, cols)
    return _add_diffs(out, ["seen", "ko", "hp_total", "statused", "lost_est"])


# --- Trends over the turns ------------------------------------------------------------------

def trend_features(timeline, n_turns=N_TURNS):
    """Total-HP difference along the battle and its slope, KO timing per side.

    Turns are counted back from the last one (``n_turns``): ``trend_hp_diff_t25`` is the difference
    5 turns before the end. With the default 30 turns the names match the actual turns.
    """
    tl = timeline[["battle_id", "turn", "side", "name", "hp_pct", "status"]].copy()
    # Each observation changes the side total by (current HP - last known HP of that Pokémon)
    prev_hp = tl.groupby(["battle_id", "side", "name"])["hp_pct"].shift().fillna(1.0)
    tl["hp_total"] = TEAM_SIZE + (tl["hp_pct"] - prev_hp).groupby([tl["battle_id"], tl["side"]]).cumsum()
    hp = tl.pivot_table(index="battle_id", columns=["side", "turn"], values="hp_total")
    diff = hp["p1"] - hp["p2"]

    out = pd.DataFrame(index=diff.index)
    shift = n_turns - N_TURNS
    for t in (10, 20, 25):
        out[f"trend_hp_diff_t{t}"] = diff[t + shift]
    out["trend_hp_slope_last5"] = (diff[n_turns] - diff[n_turns - 5]) / 5
    out["trend_hp_slope_last10"] = (diff[n_turns] - diff[n_turns - 10]) / 10
    out["trend_hp_diff_min"] = diff.min(axis=1)
    out["trend_hp_diff_max"] = diff.max(axis=1)
    out["trend_hp_diff_mean"] = diff.mean(axis=1)

    # First turn each Pokémon is seen fainted -> KO counts up to turn 20 and first-KO turn
    ko = tl[tl["status"] == "fnt"].groupby(["battle_id", "side", "name"])["turn"].min().reset_index()
    for side in ("p1", "p2"):
        k = ko[ko["side"] == side].groupby("battle_id")["turn"]
        out[f"{side}_ko_by_t20"] = k.apply(lambda s: (s <= 20 + shift).sum()).reindex(out.index).fillna(0)
        out[f"{side}_first_ko_turn"] = k.min().reindex(out.index).fillna(n_turns + 1)
    return out


# --- Actions, moves and damage --------------------------------------------------------------

def action_features(timeline, n_turns=N_TURNS):
    """Counts of actions, move categories, move groups and expected damage dealt per side."""
    tl = timeline
    keys = [tl["battle_id"], tl["side"]]
    per_side = pd.crosstab(keys, tl["action"]).rename(columns=ACTIONS)
    per_side = per_side.reindex(columns=list(ACTIONS.values()), fill_value=0)

    cats = pd.crosstab(keys, tl["move_category"]).add_prefix("n_")
    per_side = per_side.join(cats.reindex(columns=["n_physical", "n_special", "n_status"], fill_value=0))
    per_side["n_priority_moves"] = (tl["move_priority"] > 0).groupby(keys).sum()
    for group, moves in MOVE_GROUPS.items():
        per_side[f"n_{group}"] = tl["move"].isin(moves).groupby(keys).sum()

    dmg = tl["expected_damage"].fillna(0.0)
    per_side["exp_dmg_total"] = dmg.groupby(keys).sum()
    per_side["exp_dmg_last10"] = dmg.where(tl["turn"] > n_turns - 10, 0.0).groupby(keys).sum()
    attacking = tl["move_category"].isin(["physical", "special"])
    per_side["exp_dmg_per_attack"] = (dmg.where(attacking).groupby(keys).mean()).fillna(0.0)

    per_side.index.names = ["battle_id", "side"]
    per_side = per_side.reset_index()
    cols = [c for c in per_side.columns if c not in ("battle_id", "side")]
    out = _wide(per_side, cols).fillna(0)
    return _add_diffs(out, ["voluntary_switch", "n_status", "n_setup_moves", "exp_dmg_total",
                            "exp_dmg_last10", "exp_dmg_per_attack"])


def status_effect_features(timeline):
    """Turns spent with each status and effect, per side (active Pokémon)."""
    tl = timeline
    keys = [tl["battle_id"], tl["side"]]
    per_side = pd.DataFrame({f"turns_{name}": tl["status"].isin(codes).groupby(keys).sum()
                             for name, codes in STATUSES.items()})
    for e in EFFECTS:
        per_side[f"turns_{e}"] = (tl["effect"] == e).groupby(keys).sum()
    per_side.index.names = ["battle_id", "side"]
    per_side = per_side.reset_index()
    cols = [c for c in per_side.columns if c not in ("battle_id", "side")]
    out = _wide(per_side, cols)
    return _add_diffs(out, ["turns_slp", "turns_par", "turns_psn"])


def boost_features(timeline, n_turns=N_TURNS):
    """Boosts of the active Pokémon at the last turn and the highest boost reached, per stat and side."""
    cols = [f"boost_{b}" for b in BOOSTS]
    last = timeline[timeline["turn"] == n_turns].set_index(["battle_id", "side"])[cols]
    peak = timeline.groupby(["battle_id", "side"])[cols].max()
    per_side = last.add_suffix("_t30").join(peak.add_suffix("_max")).reset_index()
    names = [c for c in per_side.columns if c not in ("battle_id", "side")]
    return _wide(per_side, names)


# --- Matchup at turn 30 ---------------------------------------------------------------------

def _effectiveness(atk_type, def_types, chart):
    return float(np.prod([chart.get((atk_type, d), 1.0) for d in def_types])) if def_types else 1.0


def _best_stab(att_types, def_types, chart):
    """Best type multiplier the attacker gets with its own types (1.0 if types are unknown)."""
    return max((_effectiveness(t, def_types, chart) for t in att_types), default=1.0)


def _stage(s):
    """Gen 2 stat-stage multiplier."""
    return (2 + s) / 2 if s >= 0 else 2 / (2 - s)


def matchup_features(team, timeline, states, species_table, type_chart, n_turns=N_TURNS):
    """Type and speed matchup of the active Pokémon at turn 30, and team-level type coverage."""
    p1_types = {(b, n): t for b, n, t in team.loc[team["side"] == "p1", ["battle_id", "name", "types"]]
                .itertuples(index=False)}
    p1_spe = {(b, n): s for b, n, s in team.loc[team["side"] == "p1", ["battle_id", "name", "base_spe"]]
              .itertuples(index=False)}
    species = lookup_species(species_table, pd.Index(timeline["name"].unique()))
    sp_types, sp_spe = species["types"].to_dict(), species["base_spe"].to_dict()

    last = timeline[timeline["turn"] == n_turns].set_index(["battle_id", "side"])
    alive = states[~states["fainted"]].groupby(["battle_id", "side"])["name"].apply(list)
    p1_team = team[team["side"] == "p1"].groupby("battle_id")["name"].apply(list)
    fainted = states[states["fainted"]].groupby(["battle_id", "side"])["name"].apply(set)

    rows = []
    for bid in p1_team.index:
        a1, a2 = last.loc[(bid, "p1")], last.loc[(bid, "p2")]
        t1, t2 = p1_types.get((bid, a1["name"]), ()), sp_types.get(a2["name"], ())
        spe1 = p1_spe.get((bid, a1["name"]), 0) * _stage(a1["boost_spe"]) * (0.25 if a1["status"] == "par" else 1)
        spe2 = sp_spe.get(a2["name"], 0) * _stage(a2["boost_spe"]) * (0.25 if a2["status"] == "par" else 1)

        p1_dead = fainted.get((bid, "p1"), set())
        p1_alive = [n for n in p1_team[bid] if n not in p1_dead]          # includes unseen members
        p2_alive = alive.get((bid, "p2"), [])                             # seen and not fainted
        p1_alive_types = [p1_types[(bid, n)] for n in p1_alive]
        p2_alive_types = [sp_types.get(n, ()) for n in p2_alive]
        coverage = np.mean([max((_best_stab(a, d, type_chart) for a in p1_alive_types), default=0.0)
                            for d in p2_alive_types]) if p2_alive_types else np.nan
        threat = np.mean([max((_best_stab(a, d, type_chart) for a in p2_alive_types), default=0.0)
                          for d in p1_alive_types]) if p1_alive_types else np.nan

        rows.append({"battle_id": bid,
                     "m_active_stab_p1": _best_stab(t1, t2, type_chart),
                     "m_active_stab_p2": _best_stab(t2, t1, type_chart),
                     "m_active_speed_diff": spe1 - spe2,
                     "m_active_p1_faster": float(np.sign(spe1 - spe2)),
                     "m_active_hp_diff": a1["hp_pct"] - a2["hp_pct"],
                     "m_p1_coverage": coverage, "m_p2_threat": threat})
    out = pd.DataFrame(rows).set_index("battle_id")
    out["m_active_stab_diff"] = out["m_active_stab_p1"] - out["m_active_stab_p2"]
    out["m_coverage_diff"] = out["m_p1_coverage"] - out["m_p2_threat"]
    return out


# --- Team matchup after 30 turns (experiment 06) --------------------------------------------
#
# For every battle we build a small roster of the Pokémon still in play on each side, using only what
# that battle shows (HP, status, revealed moves, boosts of the active Pokémon) plus base stats and types
# from the species table (train only) and the type chart. Every P1 Pokémon is attacked by every P2 one and
# vice versa; the matrix is then summarised into fixed-size features.
#
# P2's "alive" set is not fainted AND not vanished with HP <= 0.5, the same estimate as lost_est: it does
# not change if P2's KOs were recorded differently (EDA, section 5.6), so these features stay robust.

def _revealed_attacks(timeline):
    """{(battle, side, name): {(move_type, category)}} of the damaging moves each Pokémon used.

    Hidden Power is skipped: its type is always recorded as normal, the real one is unknown.
    """
    att = timeline[timeline["move_category"].isin(["physical", "special"]) & (timeline["move"] != "hiddenpower")]
    out = {}
    for b, side, name, mtype, cat in att[["battle_id", "side", "name", "move_type", "move_category"]].itertuples(index=False):
        out.setdefault((b, side, name), set()).add((mtype, cat))
    return out


def _hit(att, dfn, chart):
    """Best type effectiveness and best damage index of ``att`` against ``dfn``.

    Uses the attacker's revealed damaging moves, or its own types when none was revealed. The damage index
    is effectiveness x STAB x attacking stat / defending stat (physical: atk/def, special: spa/spd).
    """
    options = att["attacks"] or [(t, None) for t in att["types"]] or [("notype", None)]
    best_eff, best_dmg = 0.0, 0.0
    for mtype, cat in options:
        eff = _effectiveness(mtype, dfn["types"], chart)
        stab = 1.5 if mtype in att["types"] else 1.0
        if cat == "physical":
            ratio = att["atk"] / dfn["def"]
        elif cat == "special":
            ratio = att["spa"] / dfn["spd"]
        else:
            ratio = max(att["atk"] / dfn["def"], att["spa"] / dfn["spd"])
        best_eff, best_dmg = max(best_eff, eff), max(best_dmg, eff * stab * ratio)
    return best_eff, best_dmg


def _attack_summary(attackers, defenders, chart):
    """Summary of the attackers x defenders matrix (NaN when one side has nobody left)."""
    keys = ["cover", "n_se_targets", "n_unwalled", "sweep", "pressure"]
    if not attackers or not defenders:
        return dict.fromkeys(keys, np.nan)
    hits = [[_hit(a, d, chart) for d in defenders] for a in attackers]
    eff = np.array([[h[0] for h in row] for row in hits])
    dmg = np.array([[h[1] for h in row] for row in hits])
    hp_att = np.array([a["hp"] for a in attackers])
    hp_def = np.array([d["hp"] for d in defenders])
    best_vs_def = dmg.max(axis=0)
    return {
        "cover": eff.max(axis=0).mean(),                       # best effectiveness against each defender
        "n_se_targets": int((eff.max(axis=0) >= 2).sum()),     # defenders hit super-effectively
        "n_unwalled": int((eff.min(axis=1) >= 1).sum()),       # attackers no remaining defender resists
        "sweep": (hp_att * dmg.min(axis=1)).max(),             # best attacker against its worst target
        "pressure": (hp_def * best_vs_def).sum() / hp_def.sum() if hp_def.sum() > 0 else best_vs_def.mean(),
    }


def team_matchup_features(team, timeline, states, species_table, type_chart, n_turns=N_TURNS):
    """Matchup of the teams still in play after 30 turns: type coverage, damage, speed, statuses, healers."""
    species = lookup_species(species_table, pd.Index(states["name"].unique()))
    attacks = _revealed_attacks(timeline)
    def users(mask):
        # (battle, side, name) of the Pokémon that used at least one move matching ``mask``
        return set(timeline.loc[mask, ["battle_id", "side", "name"]].itertuples(index=False, name=None))

    healers = users(timeline["move"].isin(HEALING_MOVES))
    damaging = timeline["move_category"].isin(["physical", "special"])
    priority_users = users(damaging & (timeline["move_priority"] > 0))     # ExtremeSpeed, Quick Attack, Mach Punch
    counter_users = users(timeline["move"].isin(COUNTER_MOVES))
    last = (timeline[timeline["turn"] == n_turns].set_index(["battle_id", "side"])
            [["name"] + [f"boost_{k}" for k in BOOSTS]].to_dict("index"))
    switches = (timeline[(timeline["action"] == "voluntary switch") & (timeline["turn"] > n_turns - 10)]
                .groupby(["battle_id", "side"]).size())
    st = states.set_index(["battle_id", "side", "name"])[["last_hp", "last_status", "fainted", "vanished"]]
    st_rows = st.to_dict("index")

    def member(b, side, name, stats, types):
        s = st_rows.get((b, side, name))
        seen = s is not None
        status = s["last_status"] if seen and isinstance(s["last_status"], str) else None
        m = {"types": tuple(types), "hp": s["last_hp"] if seen else 1.0, "status": status, "seen": seen,
             "attacks": attacks.get((b, side, name), set()), "healer": (b, side, name) in healers,
             "priority": (b, side, name) in priority_users, "counter": (b, side, name) in counter_users,
             **{k: float(stats[f"base_{k}"]) for k in ("atk", "def", "spa", "spd", "spe")}}
        m["bst"] = float(sum(stats[s_] for s_ in STATS))
        active = last[(b, side)]
        if active["name"] == name:                         # boosts only exist for the active Pokémon
            for k in BOOSTS:
                m[k] *= _stage(active[f"boost_{k}"])
        if status == "par":
            m["spe"] *= 0.25
        return m

    p1_team = team[team["side"] == "p1"]
    p2_seen = states[states["side"] == "p2"]
    p2_by_battle = p2_seen.groupby("battle_id")
    rows = []
    for b, members in p1_team.groupby("battle_id"):
        p1 = [member(b, "p1", r.name, r._asdict(), r.types) for r in members.itertuples(index=False)]
        p1_alive = [m for m, f in zip(p1, members["name"]) if not st_rows.get((b, "p1", f), {}).get("fainted", False)]
        p2_alive = []
        if b in p2_by_battle.groups:
            for r in p2_by_battle.get_group(b).itertuples(index=False):
                if r.fainted or (r.vanished and r.last_hp <= 0.5):
                    continue
                sp = species.loc[r.name]
                p2_alive.append(member(b, "p2", r.name, sp, sp["types"]))

        row = {"battle_id": b}
        for side, mine, theirs in (("p1", p1_alive, p2_alive), ("p2", p2_alive, p1_alive)):
            row.update({f"{side}_tm_{k}": v for k, v in _attack_summary(mine, theirs, type_chart).items()})
            max_spe_theirs = max((m["spe"] for m in theirs), default=0.0)
            row[f"{side}_tm_max_spe"] = max((m["spe"] for m in mine), default=0.0)
            row[f"{side}_tm_n_faster"] = sum(m["spe"] > max_spe_theirs for m in mine)
            row[f"{side}_tm_hp_healthy"] = sum(m["hp"] for m in mine if m["status"] is None)
            for name, codes in STATUSES.items():
                row[f"{side}_tm_hp_{name}"] = sum(m["hp"] for m in mine if m["status"] in codes)
            row[f"{side}_tm_alive_healers"] = sum(m["healer"] for m in mine)
            row[f"{side}_tm_alive_priority"] = sum(m["priority"] for m in mine)
            row[f"{side}_tm_alive_counter"] = sum(m["counter"] for m in mine)
            row[f"{side}_tm_switch_last10"] = switches.get((b, side), 0)
        unseen = [m["bst"] for m in p1 if not m["seen"]]
        row["p1_tm_unseen_max_bst"] = max(unseen, default=0.0)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("battle_id")
    for k in ("cover", "sweep", "pressure", "n_unwalled", "max_spe", "n_faster", "hp_healthy",
              "alive_healers", "switch_last10"):
        out[f"diff_tm_{k}"] = out[f"p1_tm_{k}"] - out[f"p2_tm_{k}"]
    return out


TEAM_MATCHUP_SUBGROUPS = {
    "tm_types_damage": ("cover", "n_se_targets", "n_unwalled", "sweep", "pressure"),
    "tm_speed": ("max_spe", "n_faster"),
    "tm_status_hp": ("hp_",),
    "tm_other": ("alive_healers", "alive_priority", "alive_counter", "switch_last10", "unseen_max_bst"),
}


def team_matchup_subgroups(columns):
    """Split the team-matchup columns into the sub-groups used for the ablation."""
    return {g: [c for c in columns if any(f"_tm_{k}" in c for k in keys)]
            for g, keys in TEAM_MATCHUP_SUBGROUPS.items()}


# --- P2's hidden team (experiment 07) -------------------------------------------------------
#
# P2's team is only partly revealed in 30 turns. P1's teams of the training set are complete (6 species
# each), so they tell which species are used together. Given the P2 Pokémon revealed in a battle, the
# co-occurrence gives a probability for every species to be among P2's unrevealed Pokémon, and from it the
# expected strength, speed and type matchup of the hidden part of P2's team.
# The co-occurrence uses only species names of the training teams, never the target.

@dataclass
class TeamCooccurrence:
    """Species co-occurrence in complete teams, with per-species attributes."""

    species: list
    pos: dict              # species name -> index
    counts: np.ndarray     # number of teams containing each species
    cooc: np.ndarray       # cooc[i, j] = number of teams containing both species
    n_teams: int
    bst: np.ndarray
    spe: np.ndarray
    hit: np.ndarray        # hit[a, d] = best STAB type effectiveness of species a against species d


def fit_team_cooccurrence(team, species_table, type_chart):
    """Fit the co-occurrence on P1's complete teams. Call it on the TRAINING teams only."""
    p1 = team[team["side"] == "p1"]
    names = sorted(p1["name"].unique())
    pos = {n: i for i, n in enumerate(names)}
    team_pos = {b: i for i, b in enumerate(p1["battle_id"].unique())}
    m = np.zeros((len(team_pos), len(names)))
    m[p1["battle_id"].map(team_pos).to_numpy(), p1["name"].map(pos).to_numpy()] = 1.0
    sp = lookup_species(species_table, pd.Index(names))
    types = sp["types"].tolist()
    hit = np.array([[_best_stab(a, d, type_chart) for d in types] for a in types])
    return TeamCooccurrence(species=names, pos=pos, counts=m.sum(axis=0), cooc=m.T @ m, n_teams=len(team_pos),
                            bst=sp[STATS].sum(axis=1).to_numpy(), spe=sp["base_spe"].to_numpy(), hit=hit)


def _hidden_weights(model, revealed):
    """Probability of each species to be one of P2's unrevealed Pokémon, given the revealed ones."""
    idx = [model.pos[n] for n in revealed if n in model.pos and model.counts[model.pos[n]] > 0]
    if idx:
        # Mean over the revealed species of P(species in team | revealed species in team)
        w = (model.cooc[:, idx] / model.counts[idx]).mean(axis=1)
        w[idx] = 0.0                                   # a species appears at most once per team
    else:
        w = model.counts / model.n_teams               # nothing known revealed: overall frequency
    total = w.sum()
    return w / total if total > 0 else model.counts / model.counts.sum()


def hidden_team_features(team, states, species_table, model):
    """Expected strength, speed and type matchup of P2's unrevealed Pokémon, and team strength estimates."""
    species = lookup_species(species_table, pd.Index(states["name"].unique()))
    bst_of = species[STATS].sum(axis=1).to_dict()
    p2 = states[states["side"] == "p2"]
    revealed = p2.groupby("battle_id")["name"].apply(list).to_dict()
    # Same "still in play" estimate as lost_est / team_matchup: robust to how P2's KOs are recorded
    p2_alive = p2[~(p2["fainted"] | (p2["vanished"] & (p2["last_hp"] <= 0.5)))]
    p2_strength = (p2_alive["last_hp"] * p2_alive["name"].map(bst_of)).groupby(p2_alive["battle_id"]).sum()
    p1_states = states[states["side"] == "p1"].set_index(["battle_id", "name"])
    p1_hp = p1_states["last_hp"].to_dict()
    p1_fainted = p1_states["fainted"].to_dict()

    rows = []
    for b, members in team[team["side"] == "p1"].groupby("battle_id"):
        seen = revealed.get(b, [])
        k = max(TEAM_SIZE - len(seen), 0)
        w = _hidden_weights(model, seen)
        alive = members[[not p1_fainted.get((b, n), False) for n in members["name"]]]
        hp = np.array([p1_hp.get((b, n), 1.0) for n in alive["name"]])
        max_spe = alive["base_spe"].max() if len(alive) else 0.0
        a_idx = [model.pos[n] for n in alive["name"] if n in model.pos]
        exp_bst = float(w @ model.bst)
        row = {
            "battle_id": b,
            "p2_hid_exp_bst": exp_bst if k else 0.0,
            "p2_hid_exp_total_bst": k * exp_bst,
            "p2_hid_exp_spe": float(w @ model.spe) if k else 0.0,
            "p2_hid_n_faster": k * float(w[model.spe > max_spe].sum()),   # hidden P2 faster than P1's fastest
            "p2_hid_threat": np.nan, "p1_hid_cover": np.nan,
            "p1_strength_est": float((hp * alive[STATS].sum(axis=1).to_numpy()).sum()),
            "p2_strength_est": float(p2_strength.get(b, 0.0)) + k * exp_bst,
        }
        if k and a_idx:
            row["p2_hid_threat"] = float(w @ model.hit[:, a_idx].mean(axis=1))   # hidden P2 vs P1 still in play
            row["p1_hid_cover"] = float(w @ model.hit[a_idx, :].max(axis=0))     # P1 still in play vs hidden P2
        rows.append(row)

    out = pd.DataFrame(rows).set_index("battle_id")
    out["diff_strength_est"] = out["p1_strength_est"] - out["p2_strength_est"]
    out["diff_hid_cover_threat"] = out["p1_hid_cover"] - out["p2_hid_threat"]
    return out


# --- Player decisions (experiment 09) -------------------------------------------------------
#
# Everything else describes the state of the battle; these features describe how each player plays,
# which carries on into the turns we do not see. The defender of a move is the opponent's Pokémon on the
# field at the end of the same turn: switches happen before moves, so it is the one that was hit.
# None of them uses KOs, so they all stay in the robust subset.

STATUS_INFLICTING = MOVE_GROUPS["sleep_moves"] | MOVE_GROUPS["para_moves"] | MOVE_GROUPS["toxic_moves"]


def decision_quality_features(team, timeline, species_table, type_chart):
    """Type effectiveness of the attacks chosen, status moves wasted and quality of the switches, per side."""
    cols = ["battle_id", "turn", "side", "name", "status", "prev_status", "switched", "move", "move_type",
            "move_category"]
    tl = timeline[cols + ["action"]]
    opp = timeline[cols].assign(side=timeline["side"].map({"p1": "p2", "p2": "p1"}))
    opp = opp.rename(columns={c: f"opp_{c}" for c in cols[3:]})
    tl = tl.merge(opp, on=["battle_id", "turn", "side"], how="left", validate="one_to_one")

    p1_types = {(b, n): t for b, n, t in team.loc[team["side"] == "p1", ["battle_id", "name", "types"]]
                .itertuples(index=False)}
    sp_types = lookup_species(species_table, pd.Index(timeline["name"].unique()))["types"].to_dict()
    cache = {}

    def eff(move_type, b, side, name):
        types = p1_types.get((b, name), sp_types.get(name, ())) if side == "p1" else sp_types.get(name, ())
        key = (move_type, types)
        if key not in cache:
            cache[key] = _effectiveness(move_type, types, type_chart)
        return cache[key]

    other = {"p1": "p2", "p2": "p1"}
    rows = tl[["battle_id", "side"]]
    # Attacks against the Pokémon actually hit (Hidden Power skipped: its recorded type is always normal)
    attack = tl["move_category"].isin(["physical", "special"]) & (tl["move"] != "hiddenpower")
    a = tl[attack]
    a_eff = pd.Series([eff(m, b, other[s], n) for m, b, s, n in
                       a[["move_type", "battle_id", "side", "opp_name"]].itertuples(index=False)], index=a.index)
    # Status moves on a target already statused (same Pokémon as the turn before) or immune by the type chart
    st = tl[tl["move"].isin(STATUS_INFLICTING)]
    already = st["opp_prev_status"].notna() & (st["opp_prev_status"] != "fnt") & ~st["opp_switched"].astype(bool)
    immune = pd.Series([eff(m, b, other[s], n) == 0 for m, b, s, n in
                        st[["move_type", "battle_id", "side", "opp_name"]].itertuples(index=False)], index=st.index)
    wasted = already | immune
    # Voluntary switches: effectiveness of the opponent's attack of that turn against the Pokémon switched in
    sw = tl[(tl["action"] == "voluntary switch") & tl["opp_move_category"].isin(["physical", "special"])
            & (tl["opp_move"] != "hiddenpower")]
    sw_eff = pd.Series([eff(m, b, s, n) for m, b, s, n in
                        sw[["opp_move_type", "battle_id", "side", "name"]].itertuples(index=False)], index=sw.index)

    def by_side(values):
        return values.groupby([rows.loc[values.index, "battle_id"], rows.loc[values.index, "side"]])

    per_side = pd.DataFrame({
        "dq_mean_eff": by_side(a_eff).mean(),
        "dq_frac_se": by_side(a_eff >= 2).mean(),
        "dq_frac_resisted": by_side(a_eff <= 0.5).mean(),
        "dq_n_immune_hits": by_side(a_eff == 0).sum(),
        "dq_wasted_status": by_side(wasted).sum(),
        "dq_frac_wasted_status": by_side(wasted).mean(),
        "dq_switch_mean_eff": by_side(sw_eff).mean(),
        "dq_switch_frac_resisted": by_side(sw_eff <= 0.5).mean(),
        "dq_n_switch_into_se": by_side(sw_eff >= 2).sum(),
    })
    full = pd.MultiIndex.from_product([timeline["battle_id"].unique(), ["p1", "p2"]], names=["battle_id", "side"])
    per_side = per_side.reindex(full)
    for c in ("dq_n_immune_hits", "dq_wasted_status", "dq_n_switch_into_se"):
        per_side[c] = per_side[c].fillna(0)                 # counts; the rates stay NaN when undefined
    per_side = per_side.reset_index()
    out = _wide(per_side, [c for c in per_side.columns if c not in ("battle_id", "side")])
    return _add_diffs(out, ["dq_mean_eff", "dq_frac_se", "dq_frac_resisted", "dq_wasted_status",
                            "dq_switch_mean_eff", "dq_switch_frac_resisted"])


def team_key_features(team, states):
    """Team identities as text keys, for the team-strength encoding (experiment 11, ``src.team_strength``).

    ``key_p1_team``: P1's six species, sorted and joined with "|". ``key_p2_seen``: the P2 species revealed in
    the 30 turns (lead included), in the same format. Never encoded here: the encoding uses the target, so it
    is fitted inside the CV folds.
    """
    p1 = team[team["side"] == "p1"].groupby("battle_id")["name"].apply(lambda s: "|".join(sorted(s)))
    seen = pd.concat([team.loc[team["side"] == "p2", ["battle_id", "name"]],
                      states.loc[states["side"] == "p2", ["battle_id", "name"]]])
    p2 = seen.drop_duplicates().groupby("battle_id")["name"].apply(lambda s: "|".join(sorted(s)))
    return pd.DataFrame({"key_p1_team": p1, "key_p2_seen": p2})


def hazard_features(team, timeline, species_table):
    """Whether each side has Spikes on its field at the last turn of ``timeline`` (experiment 16).

    The moves are replayed in turn order: Spikes sets the opponent's side, Rapid Spin clears the user's side unless the
    opponent's active Pokémon is a Ghost type. Inside a turn P1 is replayed first, because the real order is not recorded.
    Only moves that were used are seen, and nothing is fitted on the target.
    """
    types = species_table["types"].to_dict()
    p1 = team[team["side"] == "p1"]
    p1_types = dict(zip(zip(p1["battle_id"], p1["name"]), p1["types"]))
    tl = timeline.sort_values(["battle_id", "turn", "side"])
    battle, side, name, move = (tl[c].to_numpy() for c in ("battle_id", "side", "name", "move"))
    assert (side[::2] == "p1").all() and (side[1::2] == "p2").all() and (battle[::2] == battle[1::2]).all()

    on = {}
    for i in range(0, len(tl), 2):
        state = on.setdefault(battle[i], [0, 0])
        for user, foe in ((0, 1), (1, 0)):
            if move[i + user] == "spikes":
                state[foe] = 1
            elif move[i + user] == "rapidspin":
                foe_name = name[i + foe]
                foe_types = types.get(foe_name, ()) if foe == 1 else p1_types.get((battle[i], foe_name), types.get(foe_name, ()))
                if "ghost" not in foe_types:
                    state[user] = 0
    out = pd.DataFrame.from_dict(on, orient="index", columns=["p1_spikes_on", "p2_spikes_on"]).sort_index()
    out.index.name = "battle_id"
    out["diff_spikes_on"] = out["p1_spikes_on"] - out["p2_spikes_on"]
    return out


def move_key_features(timeline):
    """Species and moves used by each side, as text keys for ``src.move_typicality`` (experiment 16).

    ``key_p1_moves`` and ``key_p2_moves``: the distinct "species/move" pairs of the side, sorted and joined with "|"
    (empty when the side used no move). Never encoded here: the counts are fitted inside the CV folds.
    """
    used = timeline.dropna(subset=["move"])
    out = {}
    for side in ("p1", "p2"):
        part = used[used["side"] == side]
        pairs = (part["name"] + "/" + part["move"]).groupby(part["battle_id"]).agg(lambda s: "|".join(sorted(set(s))))
        out[f"key_{side}_moves"] = pairs
    index = pd.Index(sorted(timeline["battle_id"].unique()), name="battle_id")
    return pd.DataFrame(out).reindex(index).fillna("")


def categorical_features(team, timeline, n_turns=N_TURNS):
    """Species names for target encoding inside the CV folds (never encoded here)."""
    last = timeline[timeline["turn"] == n_turns].pivot(index="battle_id", columns="side", values="name")
    lead = team[team["side"] == "p2"].set_index("battle_id")["name"]
    return pd.DataFrame({"cat_p2_lead": lead, "cat_p1_active": last["p1"], "cat_p2_active": last["p2"]})


# --- Assembly -------------------------------------------------------------------------------

def build_features(team, timeline, states, species_table, type_chart, team_matchup=False, team_model=None,
                   decision_quality=False, team_keys=False, n_turns=N_TURNS, hazards=False, move_keys=False):
    """Return (features, groups): one row per battle_id, and {group name: list of columns}.

    ``player_won`` is included as a column when present (train), never as a feature.
    ``team_matchup=True`` adds the experiment-06 group; it is off by default so that the submission
    notebooks of experiment 04 keep producing exactly the same features.
    ``team_model`` (from ``fit_team_cooccurrence`` on the training teams) adds the experiment-07 group
    "hidden_team". ``decision_quality=True`` adds the experiment-09 group. ``team_keys=True`` adds the text
    columns of group "team_keys" (experiment 11), to be encoded by ``TeamStrengthEncoder`` inside the pipeline.
    ``n_turns`` is the last turn of ``timeline`` and ``states`` (see ``build_features_at``).
    ``hazards=True`` adds the Spikes state of experiment 16; ``move_keys=True`` adds its text columns of group "move_keys",
    to be encoded by ``MoveTypicalityEncoder`` inside the pipeline.
    """
    parts = {
        "static": static_features(team),
        "state": state_features(states),
        "trend": trend_features(timeline, n_turns),
        "action": action_features(timeline, n_turns),
        "status_effect": status_effect_features(timeline),
        "boost": boost_features(timeline, n_turns),
        "matchup": matchup_features(team, timeline, states, species_table, type_chart, n_turns),
        **({"team_matchup": team_matchup_features(team, timeline, states, species_table, type_chart, n_turns)}
           if team_matchup else {}),
        **({"hidden_team": hidden_team_features(team, states, species_table, team_model)}
           if team_model is not None else {}),
        **({"decision_quality": decision_quality_features(team, timeline, species_table, type_chart)}
           if decision_quality else {}),
        **({"team_keys": team_key_features(team, states)} if team_keys else {}),
        **({"hazards": hazard_features(team, timeline, species_table)} if hazards else {}),
        **({"move_keys": move_key_features(timeline)} if move_keys else {}),
        "species_cat": categorical_features(team, timeline, n_turns),
    }
    groups = {name: list(df.columns) for name, df in parts.items()}
    X = pd.concat(parts.values(), axis=1).sort_index()
    target = team.drop_duplicates("battle_id").set_index("battle_id")["player_won"]
    if target.notna().all():
        X["player_won"] = target.astype(int)
    return X, groups


def build_features_at(team, timeline, species_table, type_chart, n_turns, **kwargs):
    """Features of the battles cut after ``n_turns`` turns (experiment 09, snapshots of the train battles).

    The timeline is truncated and the Pokémon states are replayed on the truncated part, so the result
    is what ``build_features`` would give if the battles had been recorded for ``n_turns`` turns only.
    ``n_turns`` must be at least 21 (the trend features look 20 turns back).
    """
    tl = truncate_timeline(timeline, n_turns)
    return build_features(team, tl, pokemon_states(tl, n_turns), species_table, type_chart,
                          n_turns=n_turns, **kwargs)


def robust_columns(columns):
    """Drop features that rely on P2's recorded KOs (unreliable, see the EDA, section 5.6)."""
    fragile = ("ko", "vanished", "lost_est")
    return [c for c in columns
            if not (any(f in c for f in fragile) and (c.startswith("p2_") or c.startswith("diff_")))]
