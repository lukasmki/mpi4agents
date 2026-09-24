from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage


class GridAgent(LLMAgent):
    """2D periodic lattice: ranks are laid out on an MPI Cartesian topology and each rank
    only ever sees its four neighbors. Positions spread (or stay local) across the grid."""

    def __init__(self, comm: MPI.Comm, model: Model, steps: int = 2):
        dims = MPI.Compute_dims(comm.Get_size(), 2)
        super().__init__(comm=comm.Create_cart(dims, periods=[True, True]), model=model)
        self.dims = dims
        self.coords = self.comm.Get_coords(self.rank)
        self.steps = steps

    def draft(self, prompt: str) -> str:
        return self.ask(
            prompt,
            "Take a clear position on the PROMPT and justify it in two or three sentences.",
        )

    def update(self, prompt: str, position: str, neighbors: dict[str, str]) -> str:
        neighbors_str = "\n\n".join(
            f"NEIGHBOR ({d}):\n\n{p}" for d, p in neighbors.items()
        )
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nYOUR POSITION:\n\n{position}\n\n{neighbors_str}",
            "Update YOUR POSITION after considering your NEIGHBORs. Change your mind only if "
            "their arguments are convincing. Reply in two or three sentences.",
        )

    def exchange(self, position: str) -> dict[str, str]:
        msg = MPIMessage(self.rank, "POSITION", position)
        neighbors = {}
        for axis, (lo, hi) in enumerate([("north", "south"), ("west", "east")]):
            # shift +1: receive from the lower neighbor; shift -1: receive from the upper neighbor
            for disp, name in [(1, lo), (-1, hi)]:
                src, dst = self.comm.Shift(axis, disp)
                neighbors[name] = self.sendrecv(dest=dst, msg=msg, source=src).payload
        return neighbors

    def run(self, prompt: str | None = None) -> str:
        position = self.draft(prompt)
        for _ in range(self.steps):
            position = self.update(prompt, position, self.exchange(position))

        cells = self.gather((self.coords, position))
        if self.rank == 0:
            return "\n\n".join(f"### Cell {tuple(c)}\n\n{p}" for c, p in cells)
        return position
