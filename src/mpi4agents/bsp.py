from collections.abc import Iterator

from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent

UNCHANGED = "UNCHANGED"


class BSPAgent(LLMAgent):
    """Bulk-synchronous parallel: in each superstep every rank computes locally from the
    previous superstep's answers, exchanges its answer with all ranks, then synchronizes.
    Stops once every rank reports that its answer did not change."""

    def __init__(self, comm: MPI.Comm, model: Model, max_supersteps: int = 3):
        super().__init__(comm=comm, model=model)
        self.max_supersteps = max_supersteps

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def revise(self, prompt: str, answer: str, others: list[str]) -> str | None:
        others_str = "\n\n".join(
            f"PEER ANSWER {i}:\n\n{a}" for i, a in enumerate(others)
        )
        output = self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nYOUR ANSWER:\n\n{answer}\n\n{others_str}",
            "Revise YOUR ANSWER using anything correct from the PEER ANSWERs. "
            "Reply with the revised answer only. "
            f"If YOUR ANSWER needs no changes, reply with exactly {UNCHANGED}.",
        )
        return None if output.strip() == UNCHANGED else output

    def irun(self, prompt: str) -> Iterator[str]:
        answer = self.draft(prompt)
        yield answer
        inbox = self.allgather(answer)

        for _ in range(self.max_supersteps):
            # compute: only use data delivered in the previous superstep
            others = [a for i, a in enumerate(inbox) if i != self.rank]
            revised = self.revise(prompt, answer, others) if others else None
            if revised is not None:
                answer = revised
                yield answer

            # communicate
            inbox = self.allgather(answer)

            # synchronize: the allreduce is the superstep barrier and the convergence check
            if self.allreduce(revised is None, op=MPI.LAND):
                break
