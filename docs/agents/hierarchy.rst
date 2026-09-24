Intermittent Hierarchy
======================

Ranks usually work as equals, exchanging answers with their ring neighbors.
Every ``period`` steps a leader briefly takes charge: it gathers every answer, summarizes
where the team agrees and disagrees, and broadcasts a directive. Then the hierarchy dissolves
again. The leader role rotates each time.

::

   peer phase:    0 ─▶ 1 ─▶ 2 ─▶ 3 ─▶ 0
   leader phase:  gather ─▶ leader ─▶ bcast directive

How it works
------------

#. Every rank drafts an answer.
#. Each step, ranks swap answers around the ring and revise, following the latest directive.
#. Every ``period`` steps, rank ``(step // period) % size`` becomes leader, gathers all answers and broadcasts a new directive.

Usage
-----

:Class: :class:`~mpi4agents.hierarchy.HierarchyAgent`
:Ranks: Any.
:Returns: Every rank returns its own answer.
:LLM calls: ``1 + steps`` per rank, plus 1 each time the rank is leader.

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``steps``
     - ``4``
     - Number of peer exchange steps.
   * - ``period``
     - ``2``
     - A leader forms every ``period`` steps.

.. code-block:: python

   from mpi4agents.hierarchy import HierarchyAgent

   agent = HierarchyAgent(MPI.COMM_WORLD, model, steps=4, period=2)
   result = agent.run(prompt)

Run the example (:file:`examples/13-hierarchy.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/13-hierarchy.py --prompt "Why is the sky blue?" --steps 4 --period 2
