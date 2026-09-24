Manager-Worker (Task Farm)
==========================

Rank 0 is the manager. It splits the prompt into subtasks and hands them out
on demand: whenever a worker finishes, it gets the next subtask. Faster workers therefore do
more of the work. Once every subtask is done, the manager combines the results.

::

                ┌──▶ 1 ──┐
   0 (manager) ─┼──▶ 2 ──┼──▶ 0 (combine)
                └──▶ 3 ──┘

How it works
------------

#. The manager asks the LLM for a list of subtasks.
#. It sends one subtask to each worker, or a ``STOP`` message (``Tag.SYS``) if there are more workers than subtasks.
#. Each time a result arrives, the manager sends that worker the next subtask, or ``STOP``.
#. The manager combines all results into the final answer.

Usage
-----

:Class: :class:`~mpi4agents.farm.FarmAgent`
:Ranks: At least 2 (1 manager and 1 or more workers). Otherwise raises ``ValueError``.
:Returns: Rank 0 returns the final answer. Workers return a log of the subtasks they completed.
:LLM calls: Manager: 2. Workers: 1 per subtask they complete.

.. code-block:: python

   from mpi4agents.farm import FarmAgent

   agent = FarmAgent(MPI.COMM_WORLD, model)
   result = agent.run(prompt)

Run the example (:file:`examples/5-farm.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/5-farm.py --prompt "Why is the sky blue?"
