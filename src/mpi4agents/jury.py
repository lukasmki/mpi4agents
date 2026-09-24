import re
from collections import Counter
from collections.abc import Iterator

from mpi4agents.base import LLMAgent


class JuryAgent(LLMAgent):
    """Voting/jury: every rank proposes an answer, every rank votes for the best answer
    other than its own, and the answer with the most votes wins (ties go to the lower rank)"""

    def propose(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def vote(self, prompt: str, candidates: list[str]) -> int:
        # jurors may not vote for themselves, unless they are the only juror
        choices = [i for i in range(len(candidates)) if i != self.rank] or [self.rank]
        choices_str = "\n\n".join(
            f"CANDIDATE {n}:\n\n{candidates[i]}" for n, i in enumerate(choices)
        )
        output = self.ask(
            f"PROMPT:\n\n\t{prompt}\n\n{choices_str}",
            "You are a juror. Pick the CANDIDATE that best answers the PROMPT. "
            "Reply with only the candidate number.",
        )
        match = re.search(r"\d+", output)
        n = int(match.group()) if match else 0
        return choices[n] if n < len(choices) else choices[0]

    def irun(self, prompt: str) -> Iterator[str]:
        candidate = self.propose(prompt)
        yield candidate
        candidates = self.allgather(candidate)
        votes = self.allgather(self.vote(prompt, candidates))

        tally = Counter(votes)
        winner = min(tally, key=lambda i: (-tally[i], i))
        if self.rank == 0:
            tally_str = ", ".join(f"rank {i}: {tally[i]}" for i in sorted(tally))
            yield f"**Winner: rank {winner}** ({tally_str})\n\n{candidates[winner]}"
        else:
            yield f"Voted for rank {votes[self.rank]}\n\n{candidate}"
