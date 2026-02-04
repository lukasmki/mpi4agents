from mpi4py import MPI
from mpi4agents.base import BaseAgent, MPIMessage


class RingAgent(BaseAgent):
    def __init__(self, rank: int, comm: MPI.Comm):
        super().__init__(rank=rank, comm=comm)

    def run(self, prompt: str | None = None):
        # get answer
        my_data = {"origin": self.rank, "prompt": prompt}

        msg = MPIMessage(self.rank, "TASK", my_data)
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
