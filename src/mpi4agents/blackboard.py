from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage, Tag


class BlackboardAgent(LLMAgent):
    """Blackboard: rank 0 owns the shared board and serves READ/POST requests. The other
    ranks are knowledge sources that asynchronously read the board and post contributions."""

    def __init__(self, comm: MPI.Comm, model: Model, rounds: int = 2):
        super().__init__(comm=comm, model=model)
        if self.size < 2:
            raise ValueError(
                "Blackboard requires at least 2 ranks (1 board, 1+ sources)"
            )
        self.rounds = rounds

    @staticmethod
    def render(board: list[tuple[int, str]]) -> str:
        return "\n\n".join(
            f"ENTRY {i} (rank {rank}):\n\n{e}" for i, (rank, e) in enumerate(board)
        )

    def contribute(self, prompt: str, board: list[tuple[int, str]]) -> str:
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nBLACKBOARD:\n\n{self.render(board) or '(empty)'}",
            "Post one new entry to the BLACKBOARD that moves the PROMPT closer to being solved: "
            "a fact, a sub-result, a correction or a next step. Do not repeat existing entries.",
        )

    def summarize(self, prompt: str, board: list[tuple[int, str]]) -> str:
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nBLACKBOARD:\n\n{self.render(board)}",
            "Using the BLACKBOARD entries, write the final answer to the PROMPT.",
        )

    def serve(self, prompt: str) -> str:
        board: list[tuple[int, str]] = []
        active = self.size - 1
        while active:
            msg = self.recv()
            if msg.kind == "READ":
                self.send(
                    dest=msg.sender, msg=MPIMessage(self.rank, "BOARD", list(board))
                )
            elif msg.kind == "POST":
                board.append((msg.sender, msg.payload))
            elif msg.kind == "DONE":
                active -= 1
        return self.summarize(prompt, board)

    def source(self, prompt: str) -> str:
        entries = []
        for _ in range(self.rounds):
            self.send(dest=0, msg=MPIMessage(self.rank, "READ", None))
            board = self.recv(source=0).payload
            entry = self.contribute(prompt, board)
            self.send(dest=0, msg=MPIMessage(self.rank, "POST", entry))
            entries.append(entry)
        self.send(dest=0, msg=MPIMessage(self.rank, "DONE", None), tag=Tag.SYS)
        return "\n\n".join(entries)

    def run(self, prompt: str) -> str:
        return self.serve(prompt) if self.rank == 0 else self.source(prompt)
