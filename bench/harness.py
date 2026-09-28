"""Benchmark the communication agents on MMLU.

Every rank runs this script. The ranks are split into groups of --group-size: each group is
one multi-agent system, and the groups share out the questions. Each question's result is
appended as one JSON line to <out>/<agent>-n<group size>-g<group>.jsonl. Questions already
in those files are skipped, so an interrupted run picks up where it stopped. Score the
results with evaluate.py.

    mpirun -n 4 uv run --group bench bench/harness.py --limit 100
"""

import json
import re
import sys
import time
import traceback
from argparse import ArgumentParser
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from dotenv import load_dotenv
from mpi4py import MPI
from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError

import mmlu
from models import MeteredModel, Usage, build_model
from mpi4agents.adversarial import AdversarialAgent
from mpi4agents.alltoall import AllToAllAgent
from mpi4agents.base import LLMAgent
from mpi4agents.blackboard import BlackboardAgent
from mpi4agents.bsp import BSPAgent
from mpi4agents.butterfly import ButterflyAgent
from mpi4agents.farm import FarmAgent
from mpi4agents.grid import GridAgent
from mpi4agents.halo import HaloAgent
from mpi4agents.hierarchy import HierarchyAgent
from mpi4agents.jury import JuryAgent
from mpi4agents.morph import MorphAgent
from mpi4agents.pipe import PipeAgent
from mpi4agents.ring import RingAgent
from mpi4agents.token_passing import TokenAgent
from mpi4agents.tree import TreeAgent

RESULTS = Path(__file__).parent / "results"

EXTRACT = (
    "The RESPONSE answers a multiple choice question with options A, B, C and D. "
    "Reply with only the letter of the option it chooses, or NONE if it does not choose one."
)


class SingleAgent(LLMAgent):
    """No-communication baseline: rank 0 answers the prompt with one LLM call, the way the
    other agents draft their first answers. Run it with --group-size 1 so no ranks sit idle."""

    def irun(self, prompt: str) -> Iterator[str]:
        yield self.ask(prompt, "Answer the PROMPT concisely.") if self.rank == 0 else ""


@dataclass(frozen=True)
class Entry:
    """How an agent is scored: the final output of rank (negative counts back from the last
    rank) is the system's answer. With vote, that output combines several answers and the
    most common one is scored."""

    cls: type[LLMAgent]
    rank: int = 0
    vote: bool = False


AGENTS = {
    "single": Entry(SingleAgent),
    "ring": Entry(RingAgent),
    "pipe": Entry(PipeAgent, rank=-1),  # the last stage holds the final answer
    "tree": Entry(TreeAgent),
    "butterfly": Entry(ButterflyAgent),
    "farm": Entry(FarmAgent),
    "halo": Entry(HaloAgent, vote=True),  # rank 0 assembles every rank's section
    "token": Entry(TokenAgent),
    "alltoall": Entry(AllToAllAgent),
    "bsp": Entry(BSPAgent),
    "grid": Entry(GridAgent, vote=True),  # rank 0 lists every cell's position
    "blackboard": Entry(BlackboardAgent),
    "jury": Entry(JuryAgent),
    "hierarchy": Entry(HierarchyAgent),
    "adversarial": Entry(AdversarialAgent),
    "morph": Entry(MorphAgent),
}


class Guarded:
    """Mixin that turns a failed LLM call into an empty reply. The error is recorded against
    the question and the communication pattern carries on, instead of the other ranks
    waiting forever on a message this rank would never send."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.errors: list[str] = []

    def ask(self, prompt: str, instructions: str) -> str:
        try:
            return super().ask(prompt, instructions)
        except AgentRunError as e:
            self.errors.append(f"{type(e).__name__}: {e}"[:500])
            return ""

    def ask_list(self, prompt: str, instructions: str) -> list[str]:
        try:
            return super().ask_list(prompt, instructions)
        except AgentRunError as e:
            self.errors.append(f"{type(e).__name__}: {e}"[:500])
            return []


def fail(message: str) -> NoReturn:
    """Stop every rank, since the others would otherwise wait on this one forever"""
    print(f"error: {message}", file=sys.stderr, flush=True)
    MPI.COMM_WORLD.Abort(2)


def prepare(args, out: Path, settings: dict) -> list[dict]:
    """Check out holds only results produced with these settings, then select the questions"""
    config = {
        "model": args.model,
        "split": args.split,
        "settings": settings,
        "llm_extract": args.llm_extract,
        "prompt": mmlu.PROMPT,
    }
    path = out / "run.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if changed := [k for k in config if saved.get(k) != config[k]]:
            fail(
                f"{path} does not match this run's {', '.join(changed)}; "
                "pass another --out"
            )
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
    return mmlu.select(mmlu.load(args.split), args.subjects, args.limit, args.seed)


def completed(out: Path, stem: str) -> set[int]:
    """Ids of the questions already recorded for stem. A line left partial by a crash is
    ended, so that the next record appended starts on a line of its own."""
    done = set()
    for path in out.glob(f"{stem}-g*.jsonl"):
        text = path.read_text()
        if text and not text.endswith("\n"):
            with path.open("a") as f:
                f.write("\n")
        for line in text.splitlines():
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                pass
    return done


def attempt(
    agent: Guarded, comm: MPI.Comm, model: MeteredModel, question: dict, trace: bool
) -> list[tuple] | None:
    """Run one question on every rank of the group. Returns each rank's outputs, usage,
    seconds and errors on the group's rank 0, and None on the other ranks."""
    agent.errors.clear()
    # start together, so no rank is sent a message for this question while still on the last
    comm.barrier()
    before = replace(model.usage)
    start = time.perf_counter()
    outputs = list(agent.irun(mmlu.prompt(question)))
    seconds = time.perf_counter() - start
    return comm.gather(
        (
            outputs if trace else outputs[-1:],
            model.usage - before,
            seconds,
            agent.errors,
        ),
        root=0,
    )


def llm_extract(extractor: Agent, response: str) -> str | None:
    """Ask the model which option a response chose, for responses the patterns cannot read"""
    if not response.strip():
        return None
    try:
        return mmlu.extract(extractor.run_sync(f"RESPONSE:\n\n{response}").output)
    except AgentRunError:
        return None


def score(
    results: list[tuple],
    question: dict,
    entry: Entry,
    extractor: Agent | None,
    trace: bool,
) -> dict[str, Any]:
    """Build the record for one question from the results gathered from every rank"""
    outputs, usages, seconds, errors = zip(*results)
    finals = [o[-1] if o else "" for o in outputs]
    response = finals[entry.rank]

    pred = mmlu.extract(response, vote=entry.vote)
    extracted_by = "regex" if pred else None
    if pred is None and extractor is not None:
        pred = llm_extract(extractor, response)
        extracted_by = "llm" if pred else None

    record = {
        "id": question["id"],
        "subject": question["subject"],
        "answer": question["answer"],
        "pred": pred,
        "correct": pred == question["answer"],
        "extracted_by": extracted_by,
        "rank_preds": [mmlu.extract(f, vote=entry.vote) for f in finals],
        "response": response,
        "usage": asdict(sum(usages, Usage())),
        "seconds": max(seconds),
        "errors": [f"rank {r}: {e}" for r, errs in enumerate(errors) for e in errs],
    }
    if trace:
        record["trace"] = list(outputs)
    return record


def main():
    parser = ArgumentParser(description="Run the communication agents on MMLU.")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=AGENTS,
        default=list(AGENTS),
        metavar="AGENT",
        help=f"agents to run, in order (default: all of {', '.join(AGENTS)})",
    )
    parser.add_argument(
        "--model",
        default="random",
        help="'random' or 'test' (both offline), a model served at --base-url, or a "
        "pydantic-ai model id such as anthropic:claude-sonnet-5 (default: random)",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible endpoint serving --model, such as http://127.0.0.1:8080",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--max-tokens", type=int, help="maximum tokens in each LLM reply"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="retries of an LLM call after a transient API error (default: 3)",
    )
    parser.add_argument(
        "--split", choices=["test", "validation", "dev"], default="test"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        metavar="SUBJECT",
        help="subjects to keep, or categories: "
        + ", ".join(repr(c) for c in mmlu.CATEGORIES),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="number of questions, sampled at random (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for sampling and ordering the questions (default: 0)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="ranks per multi-agent system; the groups share out the questions "
        "(default: all ranks)",
    )
    parser.add_argument(
        "--llm-extract",
        action="store_true",
        help="ask the model for the chosen letter when a response has no recognizable answer",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="also record every rank's intermediate answers",
    )
    parser.add_argument(
        "--out", type=Path, help="results directory (default: bench/results/<model>)"
    )
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    size = args.group_size or world.size
    if args.subjects:
        args.subjects = [s for n in args.subjects for s in mmlu.CATEGORIES.get(n, [n])]
        if unknown := sorted(set(args.subjects) - mmlu.CATEGORY.keys()):
            parser.error(f"unknown subjects: {', '.join(unknown)}")
    if size < 1 or world.size % size:
        parser.error(f"--group-size {size} does not divide the {world.size} ranks")

    load_dotenv()
    out = args.out or RESULTS / re.sub(r"[^\w.-]+", "-", args.model)
    settings = {
        k: v
        for k, v in [("temperature", args.temperature), ("max_tokens", args.max_tokens)]
        if v is not None
    }
    questions = world.bcast(prepare(args, out, settings) if world.rank == 0 else None)

    group, groups = world.rank // size, world.size // size
    comm = world.Split(group, key=world.rank)
    model = MeteredModel(
        build_model(args.model, args.base_url, seed=world.rank),
        settings or None,
        args.retries,
    )
    extractor = (
        Agent(model, instructions=EXTRACT)
        if args.llm_extract and comm.rank == 0
        else None
    )

    for name in args.agents:
        entry = AGENTS[name]
        try:
            agent = type(entry.cls.__name__, (Guarded, entry.cls), {})(comm, model)
        except ValueError as e:
            if world.rank == 0:
                print(f"skipping {name}: {e}", flush=True)
            continue

        stem = f"{name}-n{size}"
        done = world.bcast(completed(out, stem) if world.rank == 0 else None)
        remaining = [q for q in questions if q["id"] not in done]
        if world.rank == 0:
            print(f"{stem}: {len(remaining)} questions to run", flush=True)
        todo = remaining[group::groups]
        path = out / f"{stem}-g{group}.jsonl"
        correct = 0
        for i, question in enumerate(todo, 1):
            results = attempt(agent, comm, model, question, args.trace)
            if comm.rank != 0:
                continue
            record = score(results, question, entry, extractor, args.trace)
            record |= {"agent": name, "ranks": size, "model": args.model}
            with path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            correct += record["correct"]
            errors = f"  {len(record['errors'])} errors" if record["errors"] else ""
            print(
                f"[{stem} g{group}] {i}/{len(todo)}  #{question['id']}  "
                f"{question['answer']} → {record['pred'] or '-'} "
                f"{'✓' if record['correct'] else '✗'}  "
                f"{record['usage']['requests']} calls  {record['seconds']:.1f}s  "
                f"acc {correct / i:.1%}{errors}",
                flush=True,
            )

    world.barrier()
    if world.rank == 0:
        print(
            f"\nResults in {out}\n"
            f"Score them with: uv run --group bench bench/evaluate.py {out}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # a rank that stops early leaves the others waiting on it, so stop the whole job
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
