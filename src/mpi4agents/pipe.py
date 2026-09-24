from pydantic import BaseModel
from typing import Any

from mpi4py import MPI
from pydantic_ai import Agent
from pydantic_ai.models import Model

from mpi4agents.base import BaseAgent, MPIMessage


class PipeMessage(BaseModel):
    origin: int
    prompt: str | None
    context: str = ""
    response: str = ""

    def serialize(self):
        return "\n\n".join(
            [
                f"PROMPT:\n\n\t{self.prompt}",
                f"CONTEXT:\n\n{self.context}",
                f"RESPONSE:\n\n{self.response}",
            ]
        )


class PipeAgent(BaseAgent):
    def __init__(self, comm: MPI.Comm, model: Model):
        super().__init__(comm=comm)
        self.agent = Agent(model=model)

    def get_context(self, data: PipeMessage) -> str:
        result = self.agent.run_sync(
            user_prompt=data.serialize(),
            instructions=(
                "State a one sentence thesis of your answer, then ",
                "list key facts relevant to answering the PROMPT.",
            ),
        )
        return result.output

    def get_response(self, data: PipeMessage) -> str:
        result = self.agent.run_sync(
            user_prompt=data.serialize(),
            instructions=(
                "Answer the given PROMPT by adding clarifying information to the previous RESPONSE. ",
                "Use the CONTEXT to inform your changes. ",
                "If the RESPONSE is empty, give an initial answer.",
            ),
        )
        return result.output

    def run(self, prompt: str | None = None) -> str:
        # all ranks generate initial response
        data = PipeMessage(origin=self.rank, prompt=prompt)
        data.context = self.get_context(data)

        if self.rank == 0:
            # generate from prompt and forward
            data.response = self.get_response(data)
            msg = MPIMessage(self.rank, "TASK", data.model_dump())
            self.send(dest=self.rank + 1, msg=msg)

        elif self.rank < self.size - 1:
            # receive, modify, and send
            msg = self.recv(source=self.rank - 1)
            # print(f"Rank {self.rank} received data from Rank {msg.payload['origin']}")
            data.response = msg.payload["response"]
            data.response = self.get_response(data)

            msg = MPIMessage(self.rank, "TASK", data.model_dump())
            self.send(dest=self.rank + 1, msg=msg)
            # print(f"Rank {self.rank} forwarding data from {msg.payload['origin']}")
        else:
            # receive, modify, and return
            msg = self.recv(source=self.rank - 1)
            # print(f"Rank {self.rank} received data from Rank {msg.payload['origin']}")
            data.response = msg.payload["response"]
            data.response = self.get_response(data)

        return data.serialize()
