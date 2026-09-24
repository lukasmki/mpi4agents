from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent


class AllToAllAgent(LLMAgent):
    """Personalized all-to-all: every rank writes a distinct critique for every other rank's
    answer, the critiques are exchanged with alltoall, and each rank revises from its own"""

    def __init__(self, comm: MPI.Comm, model: Model, rounds: int = 1):
        super().__init__(comm=comm, model=model)
        self.rounds = rounds

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def critique(self, prompt: str, own: str, other: str) -> str:
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nYOUR ANSWER:\n\n{own}\n\nTHEIR ANSWER:\n\n{other}",
            "Write feedback addressed to the author of THEIR ANSWER: what is wrong or missing, "
            "and what they could take from YOUR ANSWER. Be brief and specific.",
        )

    def revise(self, prompt: str, answer: str, feedback: list[str]) -> str:
        feedback_str = "\n\n".join(
            f"FEEDBACK {i}:\n\n{f}" for i, f in enumerate(feedback)
        )
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nYOUR ANSWER:\n\n{answer}\n\n{feedback_str}",
            "Revise YOUR ANSWER using the FEEDBACK you agree with. Reply with the revised answer only.",
        )

    def run(self, prompt: str) -> str:
        answer = self.draft(prompt)
        for _ in range(self.rounds):
            answers = self.allgather(answer)
            # outbox[i] is the critique meant only for rank i
            outbox = [
                None
                if dst == self.rank
                else self.critique(prompt, answer, answers[dst])
                for dst in range(self.size)
            ]
            inbox = self.alltoall(outbox)
            feedback = [f for f in inbox if f is not None]
            if feedback:
                answer = self.revise(prompt, answer, feedback)
        return answer
