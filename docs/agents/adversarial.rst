Adversarial/Co-evolutionary
===========================

Even ranks are proposers and odd ranks are critics. Each round, a proposer
is paired with a different critic, who attacks the proposer's answer; the proposer then
revises to withstand the attack. Critics remember their past attacks and must find new
weaknesses, so the two groups improve against each other.

::

   round 0: 0 ◀─▶ 1   2 ◀─▶ 3
   round 1: 0 ◀─▶ 3   2 ◀─▶ 1

How it works
------------

#. Each proposer drafts an answer.
#. In round ``r``, proposer ``i`` is paired with critic ``(i + r) % (size / 2)``.
#. The proposer sends its answer; the critic replies with an attack and adds it to its memory.
#. The proposer revises its answer to withstand the attack.

Usage
-----

:Class: :class:`~mpi4agents.adversarial.AdversarialAgent`
:Ranks: An even number. Otherwise raises ``ValueError``.
:Returns: Proposers return their final answer. Critics return the attacks they made.
:LLM calls: Proposers: ``1 + rounds``. Critics: ``rounds``.

.. list-table:: Parameters
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``rounds``
     - ``2``
     - Number of attack and defense rounds.

.. code-block:: python

   from mpi4agents.adversarial import AdversarialAgent

   agent = AdversarialAgent(MPI.COMM_WORLD, model, rounds=2)
   result = agent.run(prompt)

Run the example (:file:`examples/14-adversarial.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/14-adversarial.py --prompt "Why is the sky blue?" --rounds 2
