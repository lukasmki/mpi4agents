from mpi4agents.base import LLMAgent, MPIMessage


class RingAgent(LLMAgent):
    """Ring: every rank drafts an answer, then for size - 1 steps passes the answer it last
    received on to rank + 1. Each rank sees every other rank's draft exactly once and
    revises its own answer after each one."""

    def draft(self, prompt: str) -> str:
        return self.ask(prompt, "Answer the PROMPT concisely.")

    def revise(self, prompt: str, answer: str, other: str) -> str:
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nYOUR ANSWER:\n\n{answer}\n\nOTHER ANSWER:\n\n{other}",
            "Revise YOUR ANSWER using anything correct from the OTHER ANSWER. "
            "Reply with the revised answer only.",
        )

    def run(self, prompt: str) -> str:
        dst = (self.rank + 1) % self.size
        src = (self.rank - 1) % self.size

        answer = self.draft(prompt)
        msg = MPIMessage(self.rank, "ANSWER", answer)
        for _ in range(self.size - 1):
            # forward the draft received last step, so drafts travel all the way around
            msg = self.sendrecv(dest=dst, msg=msg, source=src)
            answer = self.revise(prompt, answer, msg.payload)

        return answer
