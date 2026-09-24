Pipeline
========

The answer flows through the ranks in order, like an assembly line. Rank 0
writes the first answer and each later rank refines it before passing it on. Every stage
prepares its own context up front, in parallel, so only the refinement step waits on the
previous stage.

::

   0 ──▶ 1 ──▶ 2 ──▶ 3  (final answer)

How it works
------------

#. Every rank gathers context for the prompt (a thesis and key facts) at the same time.
#. Rank 0 writes an initial answer and sends it to rank 1.
#. Each later rank waits for the answer from ``rank - 1``, refines it with its context and sends it to ``rank + 1``.

Usage
-----

:Class: :class:`~mpi4agents.pipe.PipeAgent`
:Ranks: Any.
:Returns: Each rank returns the answer as it left that stage. The last rank holds the final answer.
:LLM calls: 2 per rank.

.. code-block:: python

   from mpi4agents.pipe import PipeAgent

   agent = PipeAgent(MPI.COMM_WORLD, model)
   result = agent.run(prompt)

Run the example (:file:`examples/2-pipe.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/2-pipe.py --prompt "Why is the sky blue?"
