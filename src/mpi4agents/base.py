from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mpi4py import MPI
from mpi4py.MPI import ANY_TAG
from pydantic_ai import Agent
from pydantic_ai.models import Model


class Tag(Enum):
    """MPI message tags

    Attributes:
        MSG: agent messages carrying data
        SYS: system events, such as telling a worker to stop
        ANY: matches any tag when receiving
    """

    MSG = 1
    SYS = 2
    ANY = ANY_TAG


@dataclass
class MPIMessage:
    """Envelope for point-to-point messages between agents

    Attributes:
        sender: rank of the sending agent
        kind: message type, used by the receiver to decide what to do (e.g. ``"TASK"``)
        payload: any picklable data
    """

    sender: int
    kind: str
    payload: Any


class BaseAgent(ABC):
    """An agent bound to one MPI rank

    Wraps the mpi4py lowercase (pickle-based) point-to-point and collective operations.
    Subclasses implement :meth:`run`, which every rank calls with the same prompt.

    Args:
        comm: communicator the agent belongs to

    Attributes:
        rank: this agent's rank in ``comm``
        size: number of agents in ``comm``
        comm: the communicator
    """

    def __init__(self, comm: MPI.Comm):
        self.rank: int = comm.Get_rank()
        self.size: int = comm.Get_size()
        self.comm: MPI.Comm = comm

    def send(self, dest: int, msg: MPIMessage, tag: Tag = Tag.MSG) -> None:
        """Send message to dest"""
        self.comm.send(msg, dest=dest, tag=tag.value)

    def recv(self, source: int = MPI.ANY_SOURCE, tag: Tag = Tag.ANY) -> MPIMessage:
        """Receive message from source"""
        return self.comm.recv(source=source, tag=tag.value)

    def sendrecv(
        self,
        dest: int,
        msg: MPIMessage,
        sendtag: Tag = Tag.MSG,
        source: int = MPI.ANY_SOURCE,
        recvtag: Tag = Tag.MSG,
    ) -> MPIMessage:
        """Send msg to dest and receive a message from source in one deadlock-free step.

        Returns ``None`` when source is ``MPI.PROC_NULL``.
        """
        return self.comm.sendrecv(
            msg,
            dest=dest,
            sendtag=sendtag.value,
            source=source,
            recvtag=recvtag.value,
        )

    def probe(self, source: int = MPI.ANY_SOURCE, tag: Tag = Tag.ANY) -> bool:
        """Check if a message is waiting."""
        return self.comm.iprobe(source=source, tag=tag.value)

    def bcast(self, obj: Any = None, root: int = 0) -> Any:
        """Broadcast obj from root to all ranks"""
        return self.comm.bcast(obj, root=root)

    def gather(self, obj: Any, root: int = 0) -> list[Any] | None:
        """Gather obj from all ranks to root (None on other ranks)"""
        return self.comm.gather(obj, root=root)

    def allgather(self, obj: Any) -> list[Any]:
        """Gather obj from all ranks to all ranks"""
        return self.comm.allgather(obj)

    def alltoall(self, objs: list[Any]) -> list[Any]:
        """Send objs[i] to rank i, receive one object from every rank"""
        return self.comm.alltoall(objs)

    def allreduce(self, obj: Any, op: MPI.Op = MPI.SUM) -> Any:
        """Reduce obj over all ranks and return the result to all ranks"""
        return self.comm.allreduce(obj, op=op)

    def barrier(self) -> None:
        """Block until all ranks reach this point"""
        self.comm.barrier()

    def run(self, prompt: str):
        """Run the agent's communication pattern. Must be called on every rank."""
        raise NotImplementedError


class LLMAgent(BaseAgent):
    """BaseAgent backed by a pydantic-ai Agent

    Args:
        comm: communicator the agent belongs to
        model: pydantic-ai model used for every LLM call on this rank

    Attributes:
        agent: the underlying ``pydantic_ai.Agent``
    """

    def __init__(self, comm: MPI.Comm, model: Model):
        super().__init__(comm=comm)
        self.agent = Agent(model=model)

    def ask(self, prompt: str, instructions: str) -> str:
        """Run a single LLM call and return its text output"""
        result = self.agent.run_sync(user_prompt=prompt, instructions=instructions)
        return result.output

    def ask_list(self, prompt: str, instructions: str) -> list[str]:
        """Run a single LLM call with structured output and return a list of strings"""
        result = self.agent.run_sync(
            user_prompt=prompt, instructions=instructions, output_type=list[str]
        )
        return result.output


class EventAgent(LLMAgent):
    """Agent that loops forever, dispatching received messages to handlers by ``kind``

    Register handlers with the :meth:`on` decorator.
    """

    def __init__(self, comm: MPI.Comm, model: Model):
        super().__init__(comm=comm, model=model)
        self.handlers = {}

    def on(self, kind):
        """Decorator registering fn as the handler for messages of this kind"""

        def decorator(fn):
            self.handlers[kind] = fn
            return fn

        return decorator

    def run(self, prompt: str):
        """Receive messages forever and call the matching handler"""
        while True:
            msg = self.recv()
            if handler := self.handlers.get(msg.kind):
                handler(msg)
