from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage


class TokenAgent(LLMAgent):
    """Token passing: a single token carrying the shared answer circulates around the ring,
    and only the rank holding it may revise the answer. Rank 0 collects it at the end."""

    def __init__(self, comm: MPI.Comm, model: Model, rounds: int = 2):
        super().__init__(comm=comm, model=model)
        self.rounds = rounds

    def revise(self, token: dict) -> str:
        history = "\n".join(f"- Rank {rank}: {note}" for rank, note in token["history"])
        return self.ask(
            "\n\n".join(
                [
                    f"PROMPT:\n\n\t{token['prompt']}",
                    f"EDIT HISTORY:\n\n{history or '(none)'}",
                    f"CURRENT ANSWER:\n\n{token['answer'] or '(empty)'}",
                ]
            ),
            "You hold the token and are the only one allowed to edit the CURRENT ANSWER. "
            "Improve it with one focused change (fix, add or clarify something). "
            "If it is empty, write an initial answer. Reply with the full revised answer only.",
        )

    def summarize(self, before: str, after: str) -> str:
        return self.ask(
            f"BEFORE:\n\n{before}\n\nAFTER:\n\n{after}",
            "Describe the change from BEFORE to AFTER in one short sentence.",
        )

    def run(self, prompt: str) -> str:
        dst = (self.rank + 1) % self.size
        src = (self.rank - 1) % self.size

        token = {"prompt": prompt, "answer": "", "history": []}
        for iround in range(self.rounds):
            # rank 0 creates the token, so it skips the first receive
            if self.rank != 0 or iround > 0:
                token = self.recv(source=src).payload

            answer = self.revise(token)
            token["history"].append(
                (self.rank, self.summarize(token["answer"], answer))
            )
            token["answer"] = answer

            self.send(dest=dst, msg=MPIMessage(self.rank, "TOKEN", token))

        if self.rank == 0:
            token = self.recv(source=src).payload

        return token["answer"]
