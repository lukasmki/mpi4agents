from mpi4py import MPI
from mpi4agents.base import BaseAgent, MPIMessage


class PipeAgent(BaseAgent):
    def __init__(self, rank: int, comm: MPI.Comm):
        super().__init__(rank=rank, comm=comm)

    def run(self, prompt: str | None = None):
        if self.rank == 0:
            # generate from prompt and forward
            my_data = {"origin": self.rank, "prompt": prompt}
            msg = MPIMessage(self.rank, "TASK", my_data)
            self.send(dest=self.rank + 1, msg=msg)
        elif self.rank == self.size - 1:
            msg = self.recv(source=self.rank - 1)
            # receive and send
            print(
                f"Rank {self.rank} received data from Rank {msg.payload['origin']}, {msg.payload['prompt']}"
            )
        else:
            # receive, modify, and send
            msg = self.recv(source=self.rank - 1)
            print(
                f"Rank {self.rank} received data from Rank {msg.payload['origin']}, {msg.payload['prompt']}"
            )
            self.send(dest=self.rank + 1, msg=msg)
            print(
                f"Rank {self.rank} forwarding data from {msg.payload['origin']}, {msg.payload['prompt']}"
            )
