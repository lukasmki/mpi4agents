Ring
====

Every rank drafts an answer, then the drafts travel around a ring. At each
step a rank passes on the draft it received in the previous step, so after ``size - 1``
steps every rank has seen every other rank's draft exactly once.

::

   0 ──▶ 1 ──▶ 2 ──▶ 3
   ▲                 │
   └─────────────────┘

How it works
------------

#. Each rank drafts an answer to the prompt.
#. For ``size - 1`` steps, each rank sends the draft it holds to ``rank + 1`` and receives one from ``rank - 1`` in a single ``sendrecv``.
#. After each receive, the rank revises its own answer using the received draft.

Usage
-----

:Class: :class:`~mpi4agents.ring.RingAgent`
:Ranks: Any.
:Returns: Every rank returns its own revised answer.
:LLM calls: ``size`` per rank.

.. code-block:: python

   from mpi4agents.ring import RingAgent

   agent = RingAgent(MPI.COMM_WORLD, model)
   result = agent.run(prompt)

Run the example (:file:`examples/1-ring.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/1-ring.py --prompt "Why is the sky blue?"
