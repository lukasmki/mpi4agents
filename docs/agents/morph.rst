Role-morphing
=============

Ranks change roles as the shared draft changes. Each round, every rank picks
the role it thinks the draft needs most: ``editor``, ``researcher``, ``critic`` or
``simplifier``. Exactly one rank becomes the editor; the others write notes in their role,
and the editor merges the notes into the next draft.

::

   choose role ─▶ allgather ─▶ assign ─▶ notes ─▶ gather to editor ─▶ bcast draft

How it works
------------

#. Each rank picks a role for the current draft, and the choices are shared with ``allgather``.
#. Every rank applies the same assignment rule: the first rank that asked for editor, in an order that rotates each round, gets it. If no rank asked, the first rank in that order becomes editor. Other ranks that asked for editor get the least-used support role.
#. The other ranks write notes in their role, and the notes are gathered to the editor.
#. The editor writes the next draft and broadcasts it.

Usage
-----

:Class: :class:`~mpi4agents.morph.MorphAgent`
:Ranks: Any.
:Returns: Rank 0 returns its role history and the final draft. Other ranks return their role history.
:LLM calls: 2 per rank per round.

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``rounds``
     - ``3``
     - Number of rounds.

.. code-block:: python

   from mpi4agents.morph import MorphAgent

   agent = MorphAgent(MPI.COMM_WORLD, model, rounds=3)
   result = agent.run(prompt)

Run the example (:file:`examples/15-morph.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/15-morph.py --prompt "Why is the sky blue?" --rounds 3
