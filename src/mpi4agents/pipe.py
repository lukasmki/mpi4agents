from mpi4agents.base import LLMAgent, MPIMessage


class PipeAgent(LLMAgent):
    """Pipeline: rank 0 writes an initial answer and each following rank refines the answer
    from the previous stage before passing it on. Every stage gathers its context up front,
    in parallel, so only the refinement is serialized. The last rank holds the final answer."""

    def context(self, prompt: str) -> str:
        return self.ask(
            prompt,
            "State a one sentence thesis of your answer, then "
            "list key facts relevant to answering the PROMPT.",
        )

    def refine(self, prompt: str, context: str, answer: str) -> str:
        return self.ask(
            "\n\n".join(
                [
                    f"PROMPT:\n\n\t{prompt}",
                    f"CONTEXT:\n\n{context}",
                    f"RESPONSE:\n\n{answer or '(empty)'}",
                ]
            ),
            "Answer the given PROMPT by adding clarifying information to the previous RESPONSE. "
            "Use the CONTEXT to inform your changes. "
            "If the RESPONSE is empty, give an initial answer.",
        )

    def run(self, prompt: str) -> str:
        # all stages prepare context before the answer reaches them
        context = self.context(prompt)

        answer = "" if self.rank == 0 else self.recv(source=self.rank - 1).payload
        answer = self.refine(prompt, context, answer)

        if self.rank < self.size - 1:
            self.send(dest=self.rank + 1, msg=MPIMessage(self.rank, "ANSWER", answer))

        return answer
