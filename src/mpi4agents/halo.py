from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage


class HaloAgent(LLMAgent):
    """1D stencil over a document: each rank owns one section and repeatedly revises it
    against the halo (the neighboring sections) received from rank - 1 and rank + 1"""

    def __init__(self, comm: MPI.Comm, model: Model, steps: int = 2):
        super().__init__(comm=comm, model=model)
        self.steps = steps
        self.left = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.right = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL

    def outline(self, prompt: str) -> list[str]:
        items = self.ask_list(
            prompt,
            f"Write an outline of exactly {self.size} section titles for a response to the PROMPT. "
            "Reply with a list of titles and nothing else.",
        )
        titles = items[: self.size]
        return titles + [f"Section {i + 1}" for i in range(len(titles), self.size)]

    def write(
        self,
        prompt: str,
        titles: list[str],
        left: str | None,
        right: str | None,
        current: str,
    ) -> str:
        outline_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
        return self.ask(
            "\n\n".join(
                [
                    f"PROMPT:\n\n\t{prompt}",
                    f"OUTLINE:\n\n{outline_str}",
                    f"PREVIOUS SECTION:\n\n{left or '(none, this is the first section)'}",
                    f"YOUR SECTION: {titles[self.rank]}\n\n{current or '(not written yet)'}",
                    f"NEXT SECTION:\n\n{right or '(none, this is the last section)'}",
                ]
            ),
            "Write or revise only YOUR SECTION, one or two paragraphs. It must follow on from the "
            "PREVIOUS SECTION and lead into the NEXT SECTION without repeating them.",
        )

    def exchange(self, section: str) -> tuple[str | None, str | None]:
        msg = MPIMessage(self.rank, "HALO", section)
        from_left = self.sendrecv(dest=self.right, msg=msg, source=self.left)
        from_right = self.sendrecv(dest=self.left, msg=msg, source=self.right)
        return (
            from_left.payload if from_left else None,
            from_right.payload if from_right else None,
        )

    def run(self, prompt: str | None = None) -> str:
        titles = self.bcast(self.outline(prompt) if self.rank == 0 else None)

        section = self.write(prompt, titles, None, None, "")
        for _ in range(self.steps):
            left, right = self.exchange(section)
            section = self.write(prompt, titles, left, right, section)

        sections = self.gather(section)
        if self.rank == 0:
            return "\n\n".join(f"## {t}\n\n{s}" for t, s in zip(titles, sections))
        return f"## {titles[self.rank]}\n\n{section}"
