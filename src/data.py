"""Data loading helpers."""
import json
from pathlib import Path

# Kaggle first, then the local repo layout (run from the repo root or from notebooks/)
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/competitions/fds-pokemon-battles-prediction-2026"),
    Path("data/raw"),
    Path("../data/raw"),
]


def find_data_dir():
    """Return the first candidate directory that contains train.jsonl."""
    for path in DATA_DIR_CANDIDATES:
        if (path / "train.jsonl").exists():
            return path
    raise FileNotFoundError(f"train.jsonl not found in any of {DATA_DIR_CANDIDATES}")


def load_type_chart(path):
    """Gen 2 type chart as {(attacking_type, defending_type): multiplier}; missing pairs are 1.0."""
    chart = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            atk, dfn, mult = line.strip().split(",")
            chart[(atk, dfn)] = float(mult)
    return chart


def load_jsonl(path):
    """Load a .jsonl file into a list of dicts (one battle per line)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
