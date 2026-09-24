Epoch/Bulk-Synchronous
======================

Work proceeds in supersteps. In each one, every rank revises its answer using
only the answers delivered in the previous superstep, then all answers are exchanged, then
all ranks synchronize. The run stops early once every rank reports that its answer did not
change.

::

   compute ─▶ communicate (allgather) ─▶ sync (allreduce) ─▶ next superstep

How it works
------------

#. Every rank drafts an answer, and the drafts are shared with ``allgather``.
#. Compute: each rank revises its answer from its peers' answers, or replies ``UNCHANGED``.
#. Communicate: the new answers are shared with ``allgather``.
#. Synchronize: an ``allreduce`` with logical AND checks whether every rank was unchanged. If so, the run stops.

Usage
-----

:Class: :class:`~mpi4agents.bsp.BSPAgent`
:Ranks: Any.
:Returns: Every rank returns its own answer.
:LLM calls: ``1 + supersteps run`` per rank (at most ``1 + max_supersteps``).

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``max_supersteps``
     - ``3``
     - Maximum number of supersteps before stopping.

.. code-block:: python

   from mpi4agents.bsp import BSPAgent

   agent = BSPAgent(MPI.COMM_WORLD, model, max_supersteps=3)
   result = agent.run(prompt)

Run the example (:file:`examples/9-bsp.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/9-bsp.py --prompt "Why is the sky blue?" --max-supersteps 3
