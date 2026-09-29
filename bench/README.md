# MMLU benchmark

Runs each communication agent on [MMLU](https://arxiv.org/abs/2009.03300) (57 subjects,
14,042 test questions) and compares it with `single`, a baseline that makes one LLM call per
question and doesn't communicate.

| File | Purpose |
| --- | --- |
| [harness.py](harness.py) | MPI runner: runs agents over the questions and writes one JSON record per question |
| [evaluate.py](evaluate.py) | Scores the records: accuracy, confidence intervals, change vs. baseline, cost |
| [mmlu.py](mmlu.py) | Dataset download, prompt, answer extraction, subject categories |
| [models.py](models.py) | Model setup, usage metering and retries, offline `random` model |

## Setup

```sh
uv sync --group bench
```

On first use, the harness downloads the split from the Hugging Face
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) dataset into `bench/data/`
(3.5 MB for `test`).

If linking to Cray MPICH on Perlmutter, also run this command to use `srun` instead of `mpirun`

```sh
MPICC="cc -shared" uv pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

## Running

Every command runs under `mpirun`, and every rank runs the same script. The default `--model
random` works offline and answers each prompt with a random letter. Use it to check the
setup: every agent should score about 25%.

```sh
mpirun -n 4 uv run --group bench bench/harness.py --limit 50
```

To use a model served by an OpenAI-compatible server (llama.cpp, vLLM, Ollama), pass its URL
with `--base-url`:

```sh
mpirun -n 4 uv run --group bench bench/harness.py \
    --model unsloth/Qwen3-0.6B-GGUF --base-url http://127.0.0.1:8080 --limit 500
```

Any [pydantic-ai model id](https://ai.pydantic.dev/models/) also works. Put its API key in
`.env`:

```sh
mpirun -n 8 uv run --group bench bench/harness.py \
    --model anthropic:claude-haiku-4-5 --temperature 0 --group-size 4 --limit 500
```

By default the harness runs every agent in turn. `--agents ring tree jury` runs a subset.
`--subjects` limits the questions to given subjects or categories (`STEM`, `Humanities`,
`'Social Sciences'`, `Other`).

### Groups

`--group-size k` splits the ranks into groups of `k`. Each group is one multi-agent system
with `k` agents, and the groups divide the questions between them. For example, `-n 8
--group-size 4` runs two 4-agent systems side by side. If you leave it out, all ranks form
one group.

The harness skips an agent when the group size doesn't suit its pattern. `butterfly` needs a
power of two, `farm` and `blackboard` need at least 2, and `adversarial` needs an even
number. `single` only uses its group's rank 0, so run it on its own with `--group-size 1`:

```sh
mpirun -n 4 uv run --group bench bench/harness.py --agents single --group-size 1 --limit 500
```

### Sample and resume

`--limit N` takes the first `N` questions of a shuffle seeded with `--seed`. The questions
also run in that shuffled order. As a result, a partial run is still a uniform sample, and a
larger limit extends a smaller one.

The harness appends each result as soon as the question finishes. When you run the same
command again, it skips the questions that already have results, so an interrupted run
resumes where it stopped. Running with a larger `--limit` adds only the new questions.

## Scoring

```sh
uv run --group bench bench/evaluate.py bench/results/random
```

The script prints a summary table and a per-category table:

- **Acc %** and **95% CI**: accuracy with its Wilson interval.
- **Δ vs single**: paired accuracy difference from the baseline on the same questions, in
  percentage points, with its 95% interval.
- **No answer %**: responses with no answer letter the extractor could find. They count as
  wrong.
- **Errors %**: questions where at least one LLM call failed.
- **Calls/q**, **Tokens in/out**: mean LLM requests and tokens per question, summed over
  every rank in the group.
- **Sec/q**: mean wall time per question.

By default, every agent is scored on only the questions that all agents answered, so
unfinished runs remain comparable. Other options:

| Option | Effect |
| --- | --- |
| `--all` | Score each agent on all of its own questions |
| `--by-subject` | Add a per-subject table |
| `--agents` | Score only the named agents (the baseline is always kept) |
| `--baseline NAME` | Compare against another agent instead of `single` |
| `--csv FILE` | Also save the summary as a CSV file |

## How an answer is read

Each agent receives this prompt ([mmlu.py](mmlu.py)):

```text
The following is a multiple choice question about {subject}.

{question}

A) ...
B) ...
C) ...
D) ...

Reason briefly, then end your response with a final line of the form "Answer: X", where X is one of A, B, C or D.
```

The final output of one rank is scored as the system's answer:

| Agent | Scored output |
| --- | --- |
| `pipe` | The last rank (the last stage of the pipeline) |
| `halo` | Rank 0's assembled document, scored by majority vote over its sections |
| `grid` | Rank 0's list of every cell's position, scored by majority vote over the cells |
| All others | Rank 0 |

The extractor takes the letter from the last `Answer: X` (for `halo` and `grid`, the most
common one). If there is none, it tries looser phrasings, then `\boxed{X}`, then a response
that consists of only the letter. `<think>` blocks are ignored. With `--llm-extract`, the
harness makes one extra LLM call to read the letter from any response the rules can't parse.
That call isn't counted in the agent's usage.

## Results

Results go to `bench/results/<model>/` (change it with `--out`). Each file is named
`<agent>-n<group size>-g<group>.jsonl`, and each line holds one question:

```json
{"id": 334, "subject": "astronomy", "answer": "C", "pred": "A", "correct": false,
 "extracted_by": "regex", "rank_preds": ["A", "A", "A", "B"], "response": "...",
 "usage": {"requests": 12, "input_tokens": 1408, "output_tokens": 24}, "seconds": 8.9,
 "errors": [], "agent": "grid", "ranks": 4, "model": "random"}
```

- `rank_preds` holds the letter extracted from each rank's final output.
- `--trace` adds `trace`, which holds every intermediate answer that each rank yielded.
- `run.json` records the model, split, sampling settings, extraction mode and prompt. The
  harness refuses to add results to a directory whose `run.json` differs from the current
  run.

A failed LLM call is retried on transient errors (`--retries`, default 3). If it still
fails, the call returns an empty reply and the error goes in the record's `errors` field.
That way the other ranks aren't left waiting on a message that will never come.
