from pydantic_ai import Agent
from mpi4py.MPI import ANY_TAG
from mpi4py import MPI
from typing import Any, Annotated
from dataclasses import dataclass
from abc import ABC
from enum import Enum


class Tag(Enum):
    """MPI Message types

    Enum values:
        Tag.MSG, agent messages carrying data
        Tag.SYS, system events
        Tag.ANY, catches all events/messages
    """

    MSG = 1
    SYS = 2
    ANY = ANY_TAG


@dataclass
class MPIMessage:
    sender: int
    kind: str
    payload: Any


class BaseAgent(ABC):
    def __init__(self, rank: int, comm: MPI.Comm):
        self.rank: int = rank
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

    def run(self, prompt: str | None = None):
        raise NotImplementedError


class EventAgent(BaseAgent):
    handlers = {}

    def on(self, kind):
        def decorator(fn):
            self.handlers[kind] = fn
            return fn

        return decorator

    def run(self, prompt: str | None = None):
        while True:
            msg = self.recv()
            if handler := self.handlers.get(msg.kind):
                handler(msg)
