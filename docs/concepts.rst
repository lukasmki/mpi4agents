Concepts
========

One agent per rank
------------------

MPI starts the same program on every rank. Each rank constructs an agent from a
communicator, typically ``MPI.COMM_WORLD``, and every rank calls
:meth:`~mpi4agents.base.BaseAgent.run`. Inside ``irun``, an agent uses
:attr:`~mpi4agents.base.BaseAgent.rank` and :attr:`~mpi4agents.base.BaseAgent.size` to
decide what role it plays and which ranks it talks to.

Because every rank runs the same code, the steps that communicate must line up across
ranks. If one rank waits for a message that no other rank sends, or skips a collective
operation the others enter, the program hangs.

Class hierarchy
---------------

:class:`~mpi4agents.base.BaseAgent`
   Holds the communicator and wraps mpi4py's pickle-based operations: point-to-point
   (``send``, ``recv``, ``sendrecv``, ``probe``) and collective (``bcast``, ``gather``,
   ``allgather``, ``alltoall``, ``allreduce``, ``barrier``).

:class:`~mpi4agents.base.LLMAgent`
   Adds a pydantic-ai ``Agent``, with :meth:`~mpi4agents.base.LLMAgent.ask` for text
   output and :meth:`~mpi4agents.base.LLMAgent.ask_list` for list output. Every agent
   in :doc:`agents/index` subclasses it.

Messages
--------

Point-to-point messages are sent as an :class:`~mpi4agents.base.MPIMessage`, which records
the sender's rank, a ``kind`` string that tells the receiver what the message is (for
example ``"TASK"`` or ``"STOP"``), and a payload. Each message also has an MPI tag from
:class:`~mpi4agents.base.Tag`: ``Tag.MSG`` for data and ``Tag.SYS`` for control messages.
A receiver can filter by tag, as the task farm's manager does when it waits only for
results.

Collective operations take plain Python objects instead of ``MPIMessage``, since every rank
takes part and the sender is implied by position. For example, ``allgather(x)[i]`` is rank
``i``'s ``x``.

Writing a new agent
-------------------

Subclass :class:`~mpi4agents.base.LLMAgent`, write small methods that each make one LLM
call, and implement ``irun`` using the communication methods. ``irun`` is a generator
that yields intermediate answers as they are produced and yields the final answer last;
:meth:`~mpi4agents.base.BaseAgent.run` iterates it and returns that last value:

.. code-block:: python

   from collections.abc import Iterator

   from mpi4agents.base import LLMAgent


   class EchoAgent(LLMAgent):
       """Rank 0 answers, then broadcasts its answer to every rank"""

       def irun(self, prompt: str) -> Iterator[str]:
           answer = self.ask(prompt, "Answer concisely.") if self.rank == 0 else None
           yield self.bcast(answer)
