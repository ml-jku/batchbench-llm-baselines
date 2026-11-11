"""Zero-shot Claude Opus baseline for BatchBench compound identification.

Sends test images with "which of the following compounds has induced this
phenotype?" plus the full candidate list to the Anthropic API, saves the
predictions, and reports the token usage, scaled to the whole split. No
prices are built in: multiply the token counts by the price list yourself.

    # offline: build the request and save a preview image, nothing is sent
    python claude_opus.py --dry-run --save-preview preview.png
    # exact input tokens from Anthropic's free count_tokens endpoint
    python claude_opus.py --count-only --images-per-request 10
    # real requests (costs money): needs ANTHROPIC_API_KEY
    python claude_opus.py --effort high --images-per-request 10 -n 3
    # grow to 1000 images, reusing earlier runs (only missing images are sent)
    python claude_opus.py --effort high -n 1000 \
        --resume run_a.jsonl run_b.jsonl --output run_1000.jsonl

Images are taken in the order of a seeded permutation of the split, so the
images of a smaller -n are always part of a larger -n with the same seed.

The candidate list is the same for every image, so it goes first in the
prompt (cacheable prefix) and the image goes last.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_DATA_DIR = "../jumpcp-dataset/data/2k/500x500/final"
# Five-channel order in the arrays: DNA, RNA, ER, AGP, Mito.

QUESTION = "Which of the following compounds has induced this phenotype?"
# Recorded on every output row; kept so --resume matches the existing files.
PROVIDER = "anthropic"


@dataclass
class Usage:
    uncached: int  # input tokens that were neither written to nor read from a cache
    cache_write: int  # input tokens written to the Anthropic cache
    cache_read: int  # input tokens served from a cache
    output: int  # output tokens, including any reasoning

    @property
    def total_input(self) -> int:
        return self.uncached + self.cache_write + self.cache_read

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.uncached + other.uncached,
                     self.cache_write + other.cache_write,
                     self.cache_read + other.cache_read,
                     self.output + other.output)


def render_rgb(sample: np.ndarray) -> np.ndarray:
    """(H, W, 5) DNA/RNA/ER/AGP/Mito uint8 -> RGB: R=AGP, G=ER+Mito, B=DNA+RNA."""
    s = sample.astype(np.float32)
    rgb = np.stack([s[..., 3], s[..., 2] + s[..., 4], s[..., 0] + s[..., 1]], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def load_compounds(path: str) -> dict[int, str]:
    """label -> SMILES, from a CSV with columns smiles,label."""
    import pandas as pd

    table = pd.read_csv(path)
    table = table.sort_values("label")
    return dict(zip(table["label"].astype(int), table["smiles"].astype(str)))


def candidate_block(compounds: dict[int, str]) -> str:
    lines = "\n".join(f"{label}: {text}" for label, text in compounds.items())
    return (
        "You are shown a Cell Painting microscopy image of cells treated "
        "with one chemical compound (RGB composite: red = AGP, green = ER + "
        "Mito, blue = DNA + RNA). The compound is one of the candidates "
        f"below, given as `number: structure`.\n\n{QUESTION}\n\n"
        f"Candidates:\n{lines}"
    )


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {"type": "integer"}}},
    "required": ["answers"],
    "additionalProperties": False,
}


def answer_instruction(k: int) -> str:
    return (
        'Respond with only a JSON object of the form {"answers": [...]} '
        f"containing exactly {k} integer{'s' if k > 1 else ''}: the candidate "
        "number for each image, in the order the images were shown. Do not "
        "write any explanation or any other text. If you are unsure, still "
        "give your single best guess for every image."
    )

# def answer_instruction(k: int) -> str:
#     return (
#         'Respond with only a JSON object of the form {"answers": [...]} '
#         f"containing exactly {k} integer{'s' if k > 1 else ''}: the candidate "
#         "number for each image, in the order the images were shown. Do not "
#         "write any explanation or any other text. If you are unsure, still "
#         "make an educated guess."
#     )


def parse_answers(text: str, k: int) -> list[int | None]:
    """Parse {"answers": [...]} with exactly k integers; None where invalid."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        answers = json.loads(match.group())["answers"] if match else None
    except (ValueError, KeyError, TypeError):
        answers = None
    if (not isinstance(answers, list) or len(answers) != k
            or not all(isinstance(a, int) and not isinstance(a, bool)
                       for a in answers)):
        return [None] * k
    return answers


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def anthropic_messages(candidates: str, pngs: list[bytes]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": candidates,
        # Everything up to here is identical for every request.
        "cache_control": {"type": "ephemeral"},
    }]
    for number, png in enumerate(pngs, start=1):
        if len(pngs) > 1:
            content.append({"type": "text", "text": f"Image {number}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode(),
            },
        })
    content.append({"type": "text", "text": answer_instruction(len(pngs))})
    return [{"role": "user", "content": content}]


def ask_anthropic(client: Any, args: argparse.Namespace, candidates: str,
                  pngs: list[bytes]) -> tuple[str, Usage, str]:
    output_config: dict[str, Any] = {}
    if args.effort != "none":
        output_config["effort"] = args.effort
    if not args.free_text:
        output_config["format"] = {"type": "json_schema", "schema": ANSWER_SCHEMA}
    kwargs: dict[str, Any] = {"output_config": output_config} if output_config else {}
    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        messages=anthropic_messages(candidates, pngs),
        **kwargs,
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    usage = response.usage
    return text, Usage(
        uncached=usage.input_tokens,
        cache_write=usage.cache_creation_input_tokens or 0,
        cache_read=usage.cache_read_input_tokens or 0,
        output=usage.output_tokens,
    ), response.stop_reason


# ---------------------------------------------------------------------------
# Resume and token report
# ---------------------------------------------------------------------------


def load_previous(paths: list[str], args: argparse.Namespace, num_images: int,
                  num_candidates: int) -> dict[int, dict[str, Any]]:
    """index -> row from earlier .jsonl runs of the same setup (split, model,
    effort, group, candidate count; fields an old row lacks are not checked).

    If an index appears more than once, an answered row beats an unanswered
    one; otherwise the first file given wins.
    """
    wanted = {"split": args.split, "provider": PROVIDER,
              "model": args.model, "effort": args.effort,
              "group": args.group, "candidates": num_candidates}
    rows: dict[int, dict[str, Any]] = {}
    no_effort = 0
    for path in paths:
        with open(path) as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    # a run killed mid-write can leave a truncated last line
                    print(f"warning: {path}:{number} is not valid JSON, skipped")
                    continue
                for key, value in wanted.items():
                    if key in row and row[key] != value:
                        raise SystemExit(f"{path}:{number}: {key}={row[key]!r}, "
                                         f"but this run uses {value!r}")
                if "effort" not in row:
                    no_effort += 1
                if not 0 <= row["index"] < num_images:
                    raise SystemExit(f"{path}:{number}: index {row['index']} is "
                                     f"outside the {args.split} split")
                row.setdefault("resumed_from", str(path))
                old = rows.get(row["index"])
                if old is None or (old["prediction"] is None
                                   and row["prediction"] is not None):
                    rows[row["index"]] = row
    if no_effort:
        print(f"note: {no_effort} resumed row(s) have no recorded effort; "
              f"assuming they were run with --effort {args.effort}")
    return rows


def report(usages: list[Usage], args: argparse.Namespace, num_images: int) -> None:
    n = len(usages)
    if n == 0:
        print("\nno requests were sent")
        return
    k = args.images_per_request
    num_requests = -(-num_images // k)

    def mean(values: list[int]) -> float:
        return sum(values) / n

    uncached = mean([u.uncached for u in usages])
    written = mean([u.cache_write for u in usages])
    read = mean([u.cache_read for u in usages])
    output = mean([u.output for u in usages])
    # Largest cached prefix any request reported; what a cache hit would serve.
    prefix = max(u.cache_write + u.cache_read for u in usages)

    print(f"\n=== {args.model}, {n} request(s) ===")
    print("mean tokens per request, as reported:")
    print(f"  input, uncached     : {uncached:,.0f}")
    print(f"  input, cache write  : {written:,.0f}")
    print(f"  input, cache read   : {read:,.0f}")
    print(f"  output (incl. reasoning): {output:,.0f}")
    if prefix == 0:
        print("no cached tokens reported: either caching is off or the prefix "
              "was too short; the totals below are all uncached input")
        steady_uncached, steady_read = uncached + written + read, 0.0
    else:
        # A request that hits the cache serves the whole prefix as cache read.
        steady_read = float(prefix)
        steady_uncached = uncached + written + read - prefix
        print(f"cacheable prefix      : {prefix:,} tokens")
    condition = "if every request hits the cache" if prefix else "no caching"
    print(f"\nfull split, {num_images:,} images in {num_requests:,} requests of "
          f"{k} image(s), {condition}:")
    print(f"  input, uncached     : {steady_uncached * num_requests:,.0f}")
    print(f"  input, cache read   : {steady_read * num_requests:,.0f}")
    print(f"  output              : {output * num_requests:,.0f}")
    print("(mean of the measured requests x number of requests; plus one cache "
          "write per cache lifetime. Output varies with reasoning effort.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("-n", type=int, default=1,
                        help="images in the output, including any reused via --resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", default=None,
                        help="only draw images of this metadata group "
                             "(test split: poscon, target, Rest)")
    parser.add_argument("--balanced", action="store_true",
                        help="draw the same number of images per compound "
                             "(differs by at most one when -n doesn't divide evenly)")
    parser.add_argument("--restrict-candidates", action="store_true",
                        help="offer only the compounds that occur among the eligible "
                             "images (with --group poscon: the 8 positive controls) "
                             "instead of the full list; label numbers are kept")
    parser.add_argument("--compounds-file",
                        default=str(Path(__file__).with_name("compound.csv")),
                        help="CSV with columns smiles,label (default: compound.csv "
                             "next to this script)")
    parser.add_argument("--effort", default="low",
                        choices=["none", "low", "medium", "high", "xhigh", "max"],
                        help="Anthropic output_config.effort ('none' omits it)")
    parser.add_argument("--images-per-request", type=int, default=1,
                        help="images per request; the candidate list is then read "
                             "once per request instead of once per image")
    parser.add_argument("--free-text", action="store_true",
                        help="don't request structured JSON output (use if the "
                             "API rejects the schema); the JSON is still parsed "
                             "out of the reply")
    parser.add_argument("--max-tokens", type=int, default=5000,
                        help="output cap including any thinking")
    parser.add_argument("--dry-run", action="store_true",
                        help="no API calls; build the request and save the preview only")
    parser.add_argument("--count-only", action="store_true",
                        help="exact input tokens of one request, no generation")
    parser.add_argument("--output",
                        help="predictions .jsonl (default: predictions/"
                             "anthropic_<model>_<split>.jsonl); a .npy is saved next to it")
    parser.add_argument("--resume", nargs="+", metavar="JSONL", default=[],
                        help="earlier prediction .jsonl files to reuse: their answered "
                             "images are copied into --output and not sent again; "
                             "unanswered ones are re-asked. Must match --split, "
                             "--model and --effort. --output may be one of these files.")
    parser.add_argument("--save-preview", help="write the first RGB image here")
    args = parser.parse_args()
    if args.images_per_request < 1:
        parser.error("--images-per-request must be at least 1")

    data_dir = Path(args.data_dir).expanduser()
    images = np.load(data_dir / f"{args.split}_data_images.npy", mmap_mode="r")
    labels = np.load(data_dir / f"{args.split}_data_labels.npy")
    num_images = len(labels)

    eligible = np.ones(num_images, dtype=bool)
    if args.group is not None:
        groups = np.load(data_dir / f"{args.split}_data_metadata.npz")["group"]
        eligible = groups == args.group
        if not eligible.any():
            parser.error(f"no images with group {args.group!r}; groups in "
                         f"{args.split}: {sorted(set(groups.tolist()))}")
        print(f"group {args.group!r}: {int(eligible.sum()):,} eligible images")

    compounds = load_compounds(args.compounds_file)
    if args.restrict_candidates:
        present = set(labels[eligible].tolist())
        compounds = {label: s for label, s in compounds.items() if label in present}
    candidates = candidate_block(compounds)
    print(f"{len(compounds)} candidates, {len(candidates):,} characters")

    output = Path(args.output) if args.output else Path(__file__).with_name(
        "predictions") / f"{PROVIDER}_{args.model}_{args.split}.jsonl"
    resume_paths = {Path(p).resolve() for p in args.resume}
    if (output.exists() and output.resolve() not in resume_paths
            and not (args.dry_run or args.count_only)):
        parser.error(f"{output} already exists; pass it to --resume to extend it, "
                     "or choose another --output")

    previous = load_previous(args.resume, args, num_images, len(compounds))
    outside = [i for i in previous if not eligible[i]]
    if outside:
        parser.error(f"{len(outside)} resumed image(s) are not in group "
                     f"{args.group!r} (e.g. row {outside[0]})")
    # A fixed permutation: a smaller -n is always a prefix of a larger one.
    # Reused images count towards -n; the rest is filled from the permutation,
    # per compound when --balanced (each stratum's quota only grows with -n).
    order = [int(i) for i in np.random.default_rng(args.seed).permutation(num_images)
             if eligible[i]]
    stratum = (lambda i: int(labels[i])) if args.balanced else (lambda i: None)
    strata = sorted({stratum(i) for i in order}, key=lambda s: (s is None, s))
    base, extra = divmod(args.n, len(strata))
    need = {s: base + (rank < extra) for rank, s in enumerate(strata)}
    for i in previous:
        need[stratum(i)] -= 1
    fresh = []
    for i in order:
        if i not in previous and need[stratum(i)] > 0:
            need[stratum(i)] -= 1
            fresh.append(i)
    short = {s: c for s, c in need.items() if c > 0}
    if short:
        print(f"warning: not enough eligible images for {short} (stratum: missing)")
    # Ascending dataset order: predictions line up with the split's rows.
    selected = sorted(set(previous) | set(fresh))
    carried = {i: r for i, r in previous.items() if r["prediction"] is not None}
    todo = [i for i in selected if i not in carried]
    if args.resume:
        print(f"resume: {len(carried)} answered image(s) reused, "
              f"{len(previous) - len(carried)} unanswered re-asked, "
              f"{len(fresh)} new; {len(todo)} to send, {len(selected)} in total")
        if len(previous) > args.n:
            print(f"note: the resumed files hold {len(previous)} images, more than "
                  f"-n {args.n}; all of them are kept")
    if args.balanced:
        per_label = dict(sorted(
            Counter(int(labels[i]) for i in selected).items()))
        print(f"images per compound: {per_label}")

    k = args.images_per_request
    first_pngs = [png_bytes(render_rgb(images[i])) for i in (todo or selected)[:k]]
    if args.save_preview:
        Path(args.save_preview).write_bytes(first_pngs[0])
    print(f"image PNG size: {len(first_pngs[0]):,} bytes")

    if args.dry_run:
        return

    import anthropic

    client = anthropic.Anthropic()
    if args.count_only:
        count = client.messages.count_tokens(
            model=args.model, messages=anthropic_messages(candidates, first_pngs))
        prefix = client.messages.count_tokens(
            model=args.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": candidates}]}]).input_tokens
        print(f"input tokens, one request : {count.input_tokens:,}")
        print(f"  candidate list alone    : {prefix:,}")
        print(f"  image + instructions    : {count.input_tokens - prefix:,}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    # Written to a side file and moved over --output at the end (also on a
    # crash), so an --output that is also a --resume file is never half-written.
    partial = output.with_name(output.name + ".partial")
    # predicted[row] is the answered label for row `row` of the split, -1 if
    # that row was not sent or the answer could not be parsed.
    predicted = np.full(num_images, -1, dtype=np.int64)
    usages = []
    asked: set[int] = set()
    try:
        with partial.open("w") as log:
            for row in sorted(carried):
                predicted[row] = carried[row]["prediction"]
                log.write(json.dumps(carried[row]) + "\n")
            log.flush()
            for request, start in enumerate(range(0, len(todo), k)):
                rows = todo[start:start + k]
                pngs = [first_pngs[j] if start == 0 and j < len(first_pngs)
                        else png_bytes(render_rgb(images[row]))
                        for j, row in enumerate(rows)]
                text, usage, stop = ask_anthropic(client, args, candidates, pngs)
                answers = parse_answers(text, len(rows))
                usages.append(usage)
                for row, answer in zip(rows, answers):
                    asked.add(row)
                    if answer is not None:
                        predicted[row] = answer
                    log.write(json.dumps({
                        "index": row,  # row in {split}_data_images/labels/metadata
                        "split": args.split,
                        "label": int(labels[row]),
                        "prediction": answer,
                        "request": request,
                        "images_in_request": len(rows),
                        "raw_response": text,  # the whole request's reply
                        "stop_reason": stop,
                        "usage": asdict(usage),  # whole request, not per image
                        "provider": PROVIDER,
                        "model": args.model,
                        "effort": args.effort,
                        "group": args.group,
                        "candidates": len(compounds),
                    }) + "\n")
                    print(f"image {row}: answer={answer!r} stop={stop} "
                          f"truth={int(labels[row])} correct={answer == int(labels[row])}")
                log.flush()
                print(f"  request {request}: {usage}")
    finally:
        # Earlier unanswered rows not re-asked yet (e.g. the run stopped)
        # stay in the output so a later --resume still picks them up.
        with partial.open("a") as log:
            for row in sorted(set(previous) - set(carried) - asked):
                log.write(json.dumps(previous[row]) + "\n")
        partial.replace(output)
        np.save(output.with_suffix(".npy"), predicted)
        print(f"\nsaved {output} and {output.with_suffix('.npy')}")
        missing = len([i for i in todo if i not in asked])
        if missing:
            print(f"{missing} image(s) were not sent; to continue, run the same "
                  f"command with --resume {output}")
    report(usages, args, num_images)


if __name__ == "__main__":
    main()
