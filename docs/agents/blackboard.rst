Blackboard/Shared-state
=======================

Rank 0 owns a shared blackboard and answers ``READ`` and ``POST`` requests.
The other ranks are knowledge sources: each one reads the current board, adds an entry that
moves the problem forward, and repeats, at its own pace. When every source is done, rank 0
writes the final answer from the board.

::

   1 ─READ/POST─┐
   2 ─READ/POST─┼─▶ 0 (board)
   3 ─READ/POST─┘

How it works
------------

#. Each source sends ``READ`` and receives a copy of the board.
#. The source writes a new entry and sends it with ``POST``.
#. After ``rounds`` entries, the source sends ``DONE`` (``Tag.SYS``).
#. Once every source is done, rank 0 writes the final answer from the board.

Usage
-----

:Class: :class:`~mpi4agents.blackboard.BlackboardAgent`
:Ranks: At least 2 (1 board and 1 or more sources). Otherwise raises ``ValueError``.
:Returns: Rank 0 returns the final answer. Sources return the entries they posted.
:LLM calls: Board: 1. Sources: ``rounds``.

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``rounds``
     - ``2``
     - Number of entries each source posts.

.. code-block:: python

   from mpi4agents.blackboard import BlackboardAgent

   agent = BlackboardAgent(MPI.COMM_WORLD, model, rounds=2)
   result = agent.run(prompt)

Run the example (:file:`examples/11-blackboard.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/11-blackboard.py --prompt "Why is the sky blue?" --rounds 2
