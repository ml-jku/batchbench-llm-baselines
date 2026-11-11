"""Compare the saved predictions of every run under predictions/.

Reads every ``predictions*/*.jsonl`` written by ``claude_opus.py`` (one folder
per run, named by you) and prints accuracy, parse rate and tokens per image
for each, plus accuracy restricted to the images every run has in common so
runs on different samples are still comparable, and accuracy per image group
(poscon / target / Rest, from the split's metadata) next to the overall one.

    python compare_predictions.py
    python compare_predictions.py --root predictions/archive_runs
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# Same default as claude_opus.py, but resolved from the repo root, not the cwd.
DEFAULT_DATA_DIR = (Path(__file__).resolve().parent
                    / "../jumpcp-dataset/data/2k/500x500/final")
GROUPS = ("poscon", "target", "Rest")


def load_run(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def token_totals(rows: list[dict]) -> dict[str, int]:
    """Sum usage once per request (usage is stored on every image line)."""
    totals: Counter[str] = Counter()
    seen = set()
    for number, row in enumerate(rows):
        request = row.get("request", number)  # old files: one image per request
        if request in seen:
            continue
        seen.add(request)
        totals.update(row["usage"])
    return dict(totals)


def accuracy(rows: list[dict]) -> tuple[int, int]:
    """(correct, total); an unparsed answer counts as wrong."""
    return sum(r["prediction"] == r["label"] for r in rows), len(rows)


TEXT_COLUMNS = {"run", "model"}  # left-aligned; everything else is a number


def print_table(header: list[str], rows: list[list[str]]) -> None:
    """Print rows with every column as wide as its widest cell."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(w) if name in TEXT_COLUMNS else cell.rjust(w)
                         for name, cell, w in zip(header, cells, widths))

    print(line(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


def print_group_table(runs: dict[str, list[dict]], data_dir: Path) -> None:
    """Accuracy per image group (poscon / target / Rest) and overall, per run."""
    groups_by_split: dict[str, np.ndarray] = {}
    for split in sorted({r["split"] for rows in runs.values() for r in rows}):
        path = data_dir / f"{split}_data_metadata.npz"
        if not path.exists():
            print(f"\n(no per-group accuracy: {path} not found; pass --data-dir)")
            return
        groups_by_split[split] = np.load(path)["group"]

    def cell(rows: list[dict]) -> str:
        correct, total = accuracy(rows)
        return f"{correct / total:.2%} ({correct:,}/{total:,})" if total else "-"

    def group_of(r: dict) -> str:
        return str(groups_by_split[r["split"]][r["index"]])

    # GROUPS first, then any other group name the metadata uses
    names = list(GROUPS) + sorted({group_of(r) for rows in runs.values()
                                   for r in rows} - set(GROUPS))
    table = []
    for name, rows in runs.items():
        table.append([name] + [cell([r for r in rows if group_of(r) == g])
                               for g in names] + [cell(rows)])
    header = ["run", *names, "overall"]
    print("\naccuracy per image group (correct/images):\n")
    print_table(header, table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(Path(__file__).parent / "predictions"))
    parser.add_argument("--num-classes", type=int, default=2000)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="folder with <split>_data_metadata.npz, for the "
                             "per-group accuracy")
    args = parser.parse_args()

    paths = sorted(Path(args.root).glob("predictions*/*.jsonl"))
    if not paths:
        raise SystemExit(f"no predictions*/*.jsonl under {args.root}")
    # A run is named by its folder; the file name is added only when a folder
    # holds several runs (the model has its own column).
    per_folder = Counter(p.parent for p in paths)
    runs = {(p.parent.name if per_folder[p.parent] == 1 else f"{p.parent.name}/{p.stem}"):
            load_run(p) for p in paths}
    common = set.intersection(*({r["index"] for r in rows} for rows in runs.values()))
    chance = 1 / args.num_classes

    print(f"chance accuracy with all {args.num_classes} candidates: {chance:.4%}   "
          f"images common to all runs: {len(common)}\n")
    table = []
    for name, rows in runs.items():
        correct, total = accuracy(rows)
        c_correct, c_total = accuracy([r for r in rows if r["index"] in common])
        parsed = sum(r["prediction"] is not None for r in rows)
        distinct = len({r["prediction"] for r in rows if r["prediction"] is not None})
        # runs with a restricted candidate list record its size on every row
        candidates = rows[0].get("candidates", args.num_classes)
        table.append([
            name, rows[0].get("model", "?"), f"{total:,}", f"{parsed / total:.0%}",
            f"{candidates:,}", f"{1 / candidates:.2%}", f"{correct:,}",
            f"{correct / total:.2%}",
            f"{c_correct / c_total:.2%}" if c_total else "-", f"{distinct:,}",
        ])
    print_table(["run", "model", "images", "parsed", "candidates", "chance",
                 "correct", "acc", "acc(common)", "distinct"], table)

    print_group_table(runs, Path(args.data_dir))

    print("\ntokens per image (usage summed once per request, divided by images):\n")
    table = []
    for name, rows in runs.items():
        totals = token_totals(rows)
        n = len(rows)
        table.append([name] + [f"{totals[key] / n:,.0f}" for key in
                               ("uncached", "cache_write", "cache_read", "output")])
    print_table(["run", "uncached", "cache write", "cache read", "output"], table)

    odd = {name: Counter(r["stop_reason"] for r in rows)
           for name, rows in runs.items()}
    odd = {n: dict(c) for n, c in odd.items() if set(c) != {"end_turn"}}
    if odd:
        print("\nstop reasons other than end_turn:", odd)
    print("\n'distinct' = number of different labels predicted; a value far "
          "below the image count means the model keeps guessing the same few "
          "compounds.")


if __name__ == "__main__":
    main()
