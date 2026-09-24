Token Passing (State Passing)
=============================

A single token holding the shared answer and its edit history circulates
around the ring. Only the rank holding the token may edit the answer, so edits happen one at
a time, in order. Unlike :doc:`ring`, only one message is ever in flight.

::

   0 ──token──▶ 1 ──▶ 2 ──▶ 3
   ▲                        │
   └────────────────────────┘

How it works
------------

#. Rank 0 creates the token with an empty answer.
#. The rank holding the token makes one focused edit to the answer, records a one-sentence summary of it in the history, and sends the token to ``rank + 1``.
#. After ``rounds`` laps, the token returns to rank 0.

Usage
-----

:Class: :class:`~mpi4agents.token_passing.TokenAgent`
:Ranks: Any.
:Returns: Rank 0 returns the final answer. Other ranks return the version they last wrote.
:LLM calls: ``2 * rounds`` per rank (one to edit, one to summarize the edit).

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``rounds``
     - ``2``
     - Number of times the token goes around the ring.

.. code-block:: python

   from mpi4agents.token_passing import TokenAgent

   agent = TokenAgent(MPI.COMM_WORLD, model, rounds=2)
   result = agent.run(prompt)

Run the example (:file:`examples/7-token.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/7-token.py --prompt "Why is the sky blue?" --rounds 2
