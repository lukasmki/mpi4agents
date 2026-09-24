from collections.abc import Iterator

from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage


class HierarchyAgent(LLMAgent):
    """Intermittent hierarchy: ranks work as flat peers exchanging answers around a ring,
    and every `period` steps a (rotating) leader briefly takes charge, gathers all answers
    and broadcasts a directive before the hierarchy dissolves again"""

    def __init__(self, comm: MPI.Comm, model: Model, steps: int = 4, period: int = 2):
        super().__init__(comm=comm, model=model)
        self.steps = steps
        self.period = period

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def revise(self, prompt: str, answer: str, peer: str, directive: str) -> str:
        return self.ask(
            "\n\n".join(
                [
                    f"PROMPT:\n\n\t{prompt}",
                    f"LEADER DIRECTIVE:\n\n{directive or '(none yet)'}",
                    f"YOUR ANSWER:\n\n{answer}",
                    f"PEER ANSWER:\n\n{peer}",
                ]
            ),
            "Revise YOUR ANSWER using anything useful from the PEER ANSWER, "
            "following the LEADER DIRECTIVE. Reply with the revised answer only.",
        )

    def direct(self, prompt: str, answers: list[str]) -> str:
        answers_str = "\n\n".join(
            f"ANSWER FROM RANK {i}:\n\n{a}" for i, a in enumerate(answers)
        )
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\n{answers_str}",
            "You are the team leader for now. Summarize where the team agrees, where it "
            "disagrees, and give a short directive on what everyone should focus on next.",
        )

    def irun(self, prompt: str) -> Iterator[str]:
        dst = (self.rank + 1) % self.size
        src = (self.rank - 1) % self.size

        answer = self.draft(prompt)
        yield answer
        directive = ""
        for step in range(self.steps):
            # flat phase: peer-to-peer exchange with ring neighbors
            msg = self.sendrecv(
                dest=dst, msg=MPIMessage(self.rank, "ANSWER", answer), source=src
            )
            answer = self.revise(prompt, answer, msg.payload, directive)
            yield answer

            # hierarchical phase: a leader forms, directs, and steps down
            if (step + 1) % self.period == 0:
                leader = (step // self.period) % self.size
                answers = self.gather(answer, root=leader)
                directive = self.bcast(
                    self.direct(prompt, answers) if self.rank == leader else None,
                    root=leader,
                )
