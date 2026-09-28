"""MMLU (Hendrycks et al., 2021): loading, prompting, answer extraction and subject categories"""

import random
import re
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq

LETTERS = "ABCD"
URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/{split}-00000-of-00001.parquet"
CACHE = Path(__file__).parent / "data"

PROMPT = """\
The following is a multiple choice question about {subject}.

{question}

{options}

Reason briefly, then end your response with a final line of the form "Answer: X", \
where X is one of A, B, C or D."""

CATEGORIES = {
    "STEM": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}
CATEGORY = {subject: c for c, subjects in CATEGORIES.items() for subject in subjects}

# "Answer: B", "**Answer:** (B)", "Final answer: B"
STRICT = re.compile(r"\b(?i:answer)\s*[:：][\s*_]*\(?([A-D])\b(?![-'’])")
# "the correct answer is (B)", "the best option would be C"
LOOSE = re.compile(
    r"\b(?i:answer|option|choice)\s+(?i:is|would\s+be|should\s+be)\s*[:：]?[\s*_]*"
    r"\(?([A-D])\b(?![-'’])"
)
BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?([A-D])\b")
# the whole response is an option: "B", "(B)", "B) Paris", "B. Paris"
BARE = re.compile(r"^[\s*_#>]*\(?([A-D])(?:[).:\]]|\s*$)")
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def load(split: str = "test") -> list[dict]:
    """Download (once) and read an MMLU split from the Hugging Face ``cais/mmlu`` dataset.

    Returns one dict per question with keys ``id`` (its index in the split), ``subject``,
    ``question``, ``choices`` and ``answer`` (the correct letter).
    """
    path = CACHE / f"mmlu-{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        url = URL.format(split=split)
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        partial.rename(path)

    rows = pq.read_table(path).to_pylist()
    return [
        {
            "id": i,
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": LETTERS[row["answer"]],
        }
        for i, row in enumerate(rows)
    ]


def select(
    questions: list[dict],
    subjects: list[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Keep the questions from subjects, shuffled with seed, and truncated to limit.

    The order is random so that any prefix of a run, including one that was interrupted, is
    a uniform sample of the questions. Samples with the same seed are nested: a larger limit
    extends a smaller one.
    """
    if subjects:
        questions = [q for q in questions if q["subject"] in subjects]
    questions = list(questions)
    random.Random(seed).shuffle(questions)
    return questions[:limit]


def prompt(question: dict) -> str:
    """Format a question as the prompt every agent receives"""
    options = "\n".join(
        f"{letter}) {choice}" for letter, choice in zip(LETTERS, question["choices"])
    )
    return PROMPT.format(
        subject=question["subject"].replace("_", " "),
        question=question["question"].strip(),
        options=options,
    )


def extract(response: str, vote: bool = False) -> str | None:
    """Return the option letter a response chooses, or None if it does not choose one.

    The last explicit ``Answer: X`` wins, falling back to looser phrasings. With vote, the
    response combines several answers (such as one per section) and the most common explicit
    answer wins, ties going to the later one.
    """
    response = THINK.sub("", response)
    if letters := STRICT.findall(response):
        if not vote:
            return letters[-1]
        counts = Counter(letters)
        top = max(counts.values())
        return next(x for x in reversed(letters) if counts[x] == top)
    for pattern in (LOOSE, BOXED):
        if letters := pattern.findall(response):
            return letters[-1]
    if match := BARE.match(response):
        return match.group(1)
    return None
