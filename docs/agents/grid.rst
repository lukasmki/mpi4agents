Lattice/Grid
============

Ranks are laid out on a two-dimensional grid that wraps around at the edges
(an MPI Cartesian topology). Each rank takes a position on the prompt and only ever hears
from its four neighbors, so ideas spread across the grid, or stay local, over several
steps.

::

        north
          │
   west ─ ● ─ east
          │
        south

How it works
------------

#. The agent builds a periodic Cartesian communicator with ``comm.Create_cart``.
#. Each rank takes a position on the prompt.
#. For ``steps`` steps, each rank exchanges positions with its north, south, west and east neighbors (``Shift`` plus ``sendrecv``) and updates its own.
#. Rank 0 gathers every cell's position and grid coordinates.

Usage
-----

:Class: :class:`~mpi4agents.grid.GridAgent`
:Ranks: Any. The grid shape comes from ``MPI.Compute_dims(size, 2)``.
:Returns: Rank 0 returns every cell's final position, labelled by coordinates. Other ranks return their own position.
:LLM calls: ``1 + steps`` per rank.

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``steps``
     - ``2``
     - Number of neighbor exchange and update steps.

.. code-block:: python

   from mpi4agents.grid import GridAgent

   agent = GridAgent(MPI.COMM_WORLD, model, steps=2)
   result = agent.run(prompt)

Run the example (:file:`examples/10-grid.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/10-grid.py --prompt "Why is the sky blue?" --steps 2
