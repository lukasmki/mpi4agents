from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage


class AdversarialAgent(LLMAgent):
    """Adversarial/co-evolutionary: even ranks are proposers and odd ranks are critics.
    Each round a proposer is paired with a different critic, the critic attacks the answer,
    and the proposer revises. Critics remember past attacks and must find new weaknesses."""

    def __init__(self, comm: MPI.Comm, model: Model, rounds: int = 2):
        super().__init__(comm=comm, model=model)
        if self.size < 2 or self.size % 2:
            raise ValueError(
                f"Adversarial requires an even number of ranks, got {self.size}"
            )
        self.rounds = rounds
        self.pairs = self.size // 2
        self.is_proposer = self.rank % 2 == 0

    def opponent(self, iround: int) -> int:
        # rotate pairings so each population faces a different opponent each round
        index = self.rank // 2
        if self.is_proposer:
            return 2 * ((index + iround) % self.pairs) + 1
        return 2 * ((index - iround) % self.pairs)

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def attack(self, prompt: str, answer: str, past: list[str]) -> str:
        past_str = "\n\n".join(f"PAST ATTACK {i}:\n\n{a}" for i, a in enumerate(past))
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nANSWER:\n\n{answer}\n\n{past_str}",
            "You are an adversary. Find the most serious flaw in the ANSWER: an error, a gap, "
            "a hidden assumption or a counterexample. Do not repeat a PAST ATTACK that the "
            "ANSWER already handles. Be brief and specific.",
        )

    def defend(self, prompt: str, answer: str, attack: str) -> str:
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nYOUR ANSWER:\n\n{answer}\n\nATTACK:\n\n{attack}",
            "Revise YOUR ANSWER so it withstands the ATTACK. If the ATTACK is wrong, keep "
            "your answer and make it more convincing. Reply with the revised answer only.",
        )

    def run(self, prompt: str) -> str:
        if self.is_proposer:
            answer = self.draft(prompt)
            for iround in range(self.rounds):
                critic = self.opponent(iround)
                self.send(dest=critic, msg=MPIMessage(self.rank, "ANSWER", answer))
                attack = self.recv(source=critic).payload
                answer = self.defend(prompt, answer, attack)
            return answer

        attacks: list[str] = []
        for iround in range(self.rounds):
            proposer = self.opponent(iround)
            answer = self.recv(source=proposer).payload
            attack = self.attack(prompt, answer, attacks)
            self.send(dest=proposer, msg=MPIMessage(self.rank, "ATTACK", attack))
            attacks.append(attack)
        return "\n\n".join(f"**Attack {i}:** {a}" for i, a in enumerate(attacks))
