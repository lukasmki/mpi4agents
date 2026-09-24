from collections.abc import Iterator

from mpi4agents.base import LLMAgent, MPIMessage


class TreeAgent(LLMAgent):
    """Binary tree reduction: children send answers up, parents merge, root holds the final answer"""

    def parent(self) -> int | None:
        return (self.rank - 1) // 2 if self.rank > 0 else None

    def children(self) -> list[int]:
        return [c for c in (2 * self.rank + 1, 2 * self.rank + 2) if c < self.size]

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def merge(self, prompt: str, answers: list[str]) -> str:
        answers_str = "\n\n".join(f"ANSWER {i}:\n\n{a}" for i, a in enumerate(answers))
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\n{answers_str}",
            "Merge the ANSWERs into a single answer to the PROMPT. "
            "Keep every correct point, drop duplicates and resolve contradictions.",
        )

    def irun(self, prompt: str) -> Iterator[str]:
        answer = self.draft(prompt)
        yield answer

        # wait on the subtree below, then merge it into this rank's answer
        if children := self.children():
            answers = [answer] + [self.recv(source=c).payload for c in children]
            answer = self.merge(prompt, answers)
            yield answer

        # forward the merged subtree answer to the parent
        if (parent := self.parent()) is not None:
            self.send(dest=parent, msg=MPIMessage(self.rank, "PARTIAL", answer))
