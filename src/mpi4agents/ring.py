from typing import Any

from mpi4py import MPI
from pydantic_ai import Agent
from pydantic_ai.models import Model

from mpi4agents.base import BaseAgent, MPIMessage


class RingAgent(BaseAgent):
    def __init__(self, comm: MPI.Comm, model: Model):
        super().__init__(comm=comm)
        self.agent = Agent(model=model)

    def process(self, data: dict[str, Any]):
        pass

    def run(self, prompt: str | None = None):
        # get answer

        data = {"origin": self.rank, "prompt": prompt}

        msg = MPIMessage(self.rank, "TASK", data)
        dst = (self.rank + 1) % self.size
        src = (self.rank - 1) % self.size
        for istep in range(self.size - 1):
            print(
                f"Rank {self.rank} forwarding data from {msg.payload['origin']}, {msg.payload['prompt']}"
            )
            msg = self.sendrecv(dest=dst, msg=msg, source=src)
            # revise answer based on other answer
            print(
                f"Rank {self.rank} received data from Rank {msg.payload['origin']}, {msg.payload['prompt']}"
            )
