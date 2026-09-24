Nearest Neighbor (Halo/Stencil)
===============================

A one-dimensional stencil over a document. Rank 0 writes an outline with one
section per rank, and each rank owns one section. On each step, ranks swap sections with
their neighbors (the "halo") and revise their own section so it follows on from the previous
section and leads into the next.

::

   ┌───┐   ┌───┐   ┌───┐   ┌───┐
   │ 0 │◀─▶│ 1 │◀─▶│ 2 │◀─▶│ 3 │
   └───┘   └───┘   └───┘   └───┘

How it works
------------

#. Rank 0 writes an outline of ``size`` section titles and broadcasts it.
#. Each rank writes a first version of its own section.
#. For ``steps`` steps, each rank exchanges sections with ``rank - 1`` and ``rank + 1`` and revises its own. Ranks at the ends have no neighbor on that side (``MPI.PROC_NULL``).
#. Rank 0 gathers the sections into the full document.

Usage
-----

:Class: :class:`~mpi4agents.halo.HaloAgent`
:Ranks: Any.
:Returns: Rank 0 returns the full document. Other ranks return their own section.
:LLM calls: ``1 + steps`` per rank, plus 1 for the outline on rank 0.

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``steps``
     - ``2``
     - Number of halo exchange and revision steps.

.. code-block:: python

   from mpi4agents.halo import HaloAgent

   agent = HaloAgent(MPI.COMM_WORLD, model, steps=2)
   result = agent.run(prompt)

Run the example (:file:`examples/6-halo.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/6-halo.py --prompt "Why is the sky blue?" --steps 2
