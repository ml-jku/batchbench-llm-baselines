# BatchBench LLM baselines

This repository contains the zero-shot Claude Opus 5.5 baseline for the
[BatchBench challenge](https://huggingface.co/spaces/ai-lab-jku/jumpcp-challenge).
The model is shown a Cell Painting image and asked: *"Which of the following
compounds has induced this phenotype?"*. The prompt lists all 2000 candidate
compounds as `number: SMILES`, and the model answers with a candidate number.
Nothing is trained: the model is used as-is through the Anthropic API.

> [!IMPORTANT]
> **This code does not follow the
> submission pipeline.** It has no Docker image and no `Submission` class,
> and it needs network access to the Anthropic API, which the official
> evaluation runtime does not provide. It
> only shows how the LLM baseline on the leaderboard was computed.
>
> To build your own submission, follow the structure of the
> [BatchBench baselines](https://github.com/hiitsmeme/jumpcp-baselines)
> repository.

## How it works

- **Image:** the five channels (DNA, RNA, ER, AGP, Mito; 500x500 uint8) are
  combined into one RGB PNG: red = AGP, green = ER + Mito, blue = DNA + RNA.
- **Prompt:** the task description and the candidate list come first and
  are the same in every request, so they are marked as a cacheable prefix
  (about 97k tokens). Then come the image(s), labelled `Image 1:`, `Image 2:`,
  ... when there are several per request. The answer instruction is last.
- **Answer:** the model is asked for `{"answers": [...]}` with exactly one
  integer per image, enforced through structured output. If a reply is not
  exactly that, the whole request counts as unanswered and is scored as wrong.
- **Image order:** images are drawn from a seeded permutation of the split
  (`--seed 42`), so the images of a smaller `-n` are always part of a larger
  `-n`.

```
claude_opus.py           # query the API, save predictions, report token usage
compare_predictions.py   # accuracy / per-group accuracy / tokens of all saved runs
compound.csv             # smiles,label: the 2000 candidates (label = class index)
model_outputs/
  claude-opus-5-5_test_responses.txt  # raw model replies of the leaderboard run
predictions/
  predictions_<name>/    # one folder per finished run: anthropic_<model>_<split>.jsonl + .npy
```

[`model_outputs/claude-opus-5-5_test_responses.txt`](model_outputs/claude-opus-5-5_test_responses.txt)
holds the model's output strings of the leaderboard run, one line per request (4,615
requests of 10 images, in the order they were sent).

## Running the baseline

Run the following commands from the repository root. First, install the
dependencies and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
and set your Anthropic API key:

```bash
python -m pip install -r requirements.txt
python -m pip install -U huggingface_hub
export ANTHROPIC_API_KEY=...
```

### Downloading the data

Download the [2k classes set with 500x500 resolution](https://huggingface.co/datasets/ai-lab-jku/jumpcp-challenge-datasets/tree/main/2k/500x500),
the same setting as in the leaderboard:

```bash
hf download ai-lab-jku/jumpcp-challenge-datasets \
  --repo-type dataset \
  --include "2k/500x500/*" \
  --local-dir data
```

The public dataset contains only the train split. The leaderboard number was
computed on the held-out test split, which is not public.

### Evaluation run

Before spending money, check the setup:

```bash
# no API call: build the request and save an RGB preview of the first image
python claude_opus.py --data-dir data/2k/500x500 --split train \
    --dry-run --save-preview preview.png

# exact input tokens of one request (Anthropic count_tokens, free)
python claude_opus.py --data-dir data/2k/500x500 --split train \
    --count-only --images-per-request 10
```

The leaderboard entry uses `claude-opus-5-5` at effort `high`, 10 images per
request, all 2000 candidates, on the full test split (46,150 images). With
`--resume`, images that are already answered are reused and not sent again:

```bash
python claude_opus.py --data-dir <test-split-dir> --split test \
    --model claude-opus-5-5 --effort high --images-per-request 10 -n 1000 \
    --output predictions/predictions_1k/anthropic_claude-opus-5-5_test.jsonl

python claude_opus.py --data-dir <test-split-dir> --split test \
    --model claude-opus-5-5 --effort high --images-per-request 10 -n 46150 \
    --resume predictions/predictions_1k/anthropic_claude-opus-5-5_test.jsonl \
    --output predictions/predictions_full/anthropic_claude-opus-5-5_test.jsonl
```

To run the same setup on the public data, use `--data-dir data/2k/500x500
--split train` and an output name ending in `_train.jsonl`. If a run stops
partway, rerun the same command with its own output file added to `--resume`.

Useful options:

| Option | Meaning |
|---|---|
| `--effort {none,low,medium,high,xhigh,max}` | `output_config.effort` (default `low`) |
| `--images-per-request K` | K images share one request, so the candidate list is read once per K images |
| `--group {poscon,target,Rest}` | only draw images from this metadata group |
| `--balanced` | draw the same number of images per compound |
| `--restrict-candidates` | list only the compounds that occur among the eligible images |
| `--free-text` | don't request structured output (if the API rejects the schema) |

Each run writes `<output>.jsonl`, with one line per image (`index`, `label`,
`prediction`, `raw_response`, token `usage` and the run settings), and
`<output>.npy`, with one prediction per row of the whole split (`-1` for rows
that were not sent or could not be parsed). At the end of a run,
`claude_opus.py` prints the mean tokens per request, projected to the full
split.

## Accuracy

```bash
python compare_predictions.py --data-dir <split-dir>
```

This reads every `predictions/predictions_*/*.jsonl` and prints, per run, the
accuracy, the parse rate, how many different labels
were predicted, the accuracy per group (poscon / target / Rest, read from the
split's metadata in `--data-dir`), and the tokens per image. Use `--root` to
point it at another folder of runs.
