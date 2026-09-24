from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage


class ButterflyAgent(LLMAgent):
    """Recursive doubling: at stage k each rank merges with rank ^ 2**k, so after
    log2(size) stages every rank's answer incorporates every other rank's answer"""

    def __init__(self, comm: MPI.Comm, model: Model):
        super().__init__(comm=comm, model=model)
        if self.size & (self.size - 1):
            raise ValueError(
                f"Butterfly requires a power of two ranks, got {self.size}"
            )

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def merge(self, prompt: str, answers: list[str]) -> str:
        answers_str = "\n\n".join(f"ANSWER {i}:\n\n{a}" for i, a in enumerate(answers))
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\n{answers_str}",
            "Merge the ANSWERs into a single answer to the PROMPT. "
            "Keep every correct point, drop duplicates and resolve contradictions.",
        )

    def run(self, prompt: str | None = None) -> str:
        answer = self.draft(prompt)

        stage = 0
        while (1 << stage) < self.size:
            partner = self.rank ^ (1 << stage)
            msg = self.sendrecv(
                dest=partner,
                msg=MPIMessage(self.rank, "PARTIAL", answer),
                source=partner,
            )
            # order by rank so both partners merge the same input
            pair = (
                [answer, msg.payload] if self.rank < partner else [msg.payload, answer]
            )
            answer = self.merge(prompt, pair)
            stage += 1

        return answer
