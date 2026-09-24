Butterfly
=========

A `butterfly network <https://en.wikipedia.org/wiki/Butterfly_network>`_
(recursive doubling). At stage ``k`` each rank pairs with ``rank XOR 2**k`` and the pair
merge their answers. After ``log2(size)`` stages, every rank's answer includes every other
rank's answer, with no single rank doing all the merging.

::

   stage 0     stage 1
   0 ◀─▶ 1     0 ◀─▶ 2
   2 ◀─▶ 3     1 ◀─▶ 3

How it works
------------

#. Every rank drafts an answer.
#. At each stage, the rank swaps its answer with its partner and merges the two. Both partners merge the same pair in the same order.

Usage
-----

:Class: :class:`~mpi4agents.butterfly.ButterflyAgent`
:Ranks: A power of two. Other sizes raise ``ValueError``.
:Returns: Every rank returns a combined answer. The wording may differ between ranks because each merge is a separate LLM call.
:LLM calls: ``1 + log2(size)`` per rank.

.. code-block:: python

   from mpi4agents.butterfly import ButterflyAgent

   agent = ButterflyAgent(MPI.COMM_WORLD, model)
   result = agent.run(prompt)

Run the example (:file:`examples/4-butterfly.py`):

.. code-block:: sh

   mpirun -n 4 uv run examples/4-butterfly.py --prompt "Why is the sky blue?"
