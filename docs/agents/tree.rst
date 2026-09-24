Tree
====

Answers are merged up a binary tree. Rank ``r`` has children ``2r + 1`` and
``2r + 2``; each parent merges its children's answers into its own and passes the result up,
so rank 0 ends up with a merge of every rank's answer.

::

           0
         ╱   ╲
        1     2
       ╱ ╲   ╱ ╲
      3   4 5   6

How it works
------------

#. Every rank drafts an answer.
#. Ranks with children wait for them, then merge the children's answers with their own.
#. Every rank except rank 0 sends its (merged) answer to its parent.

Usage
-----

:Class: :class:`~mpi4agents.tree.TreeAgent`
:Ranks: Any.
:Returns: Rank 0 returns the final merged answer. Other ranks return the merged answer for their subtree.
:LLM calls: 1, plus 1 more for ranks with children.

.. code-block:: python

   from mpi4agents.tree import TreeAgent

   agent = TreeAgent(MPI.COMM_WORLD, model)
   result = agent.run(prompt)

Run the example (:file:`examples/3-tree.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/3-tree.py --prompt "Why is the sky blue?"
