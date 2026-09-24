Voting/Jury
===========

Every rank proposes an answer, then acts as a juror and votes for the best
proposal other than its own. The proposal with the most votes wins, and ties go to the lower
rank.

::

   propose ─▶ allgather candidates ─▶ vote ─▶ allgather votes ─▶ tally

How it works
------------

#. Every rank proposes an answer, and the proposals are shared with ``allgather``.
#. Each rank votes for a candidate other than itself, unless it is the only rank.
#. The votes are shared with ``allgather`` and every rank computes the same tally.

Usage
-----

:Class: :class:`~mpi4agents.jury.JuryAgent`
:Ranks: Any.
:Returns: Rank 0 returns the winner, the vote tally and the winning answer. Other ranks return their vote and their own proposal.
:LLM calls: 2 per rank.

.. code-block:: python

   from mpi4agents.jury import JuryAgent

   agent = JuryAgent(MPI.COMM_WORLD, model)
   result = agent.run(prompt)

Run the example (:file:`examples/12-jury.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/12-jury.py --prompt "Why is the sky blue?"
