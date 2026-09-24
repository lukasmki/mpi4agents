Personalized All-to-All
=======================

Every rank writes separate feedback for every other rank's answer. The
feedback is exchanged with ``alltoall``, so each rank receives only the feedback addressed to
it, then revises its answer.

::

   rank i writes outbox[j] for every j ≠ i
   alltoall: rank j receives outbox[j] from every rank

How it works
------------

#. Every rank drafts an answer, and the drafts are shared with ``allgather``.
#. Each rank writes one critique per other rank, based on that rank's draft.
#. ``alltoall`` delivers each critique to the rank it was written for.
#. Each rank revises its answer using the feedback it received.

Usage
-----

:Class: :class:`~mpi4agents.alltoall.AllToAllAgent`
:Ranks: Any.
:Returns: Every rank returns its own revised answer.
:LLM calls: ``1 + rounds * size`` per rank (``size - 1`` critiques and 1 revision per round).

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``rounds``
     - ``1``
     - Number of critique and revision rounds.

.. code-block:: python

   from mpi4agents.alltoall import AllToAllAgent

   agent = AllToAllAgent(MPI.COMM_WORLD, model, rounds=1)
   result = agent.run(prompt)

Run the example (:file:`examples/8-alltoall.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/8-alltoall.py --prompt "Why is the sky blue?" --rounds 1
