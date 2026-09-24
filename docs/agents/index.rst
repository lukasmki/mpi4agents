Agents
======

Each agent implements one communication pattern. Every rank constructs the same agent
class and calls :meth:`~mpi4agents.base.BaseAgent.run` with the same prompt. The
pattern decides which ranks talk to each other, and when.

Standard patterns
-----------------

.. toctree::
   :maxdepth: 1

   ring
   pipe
   tree
   butterfly
   farm
   halo
   token
   alltoall
   bsp

Other patterns
--------------

.. toctree::
   :maxdepth: 1

   grid
   blackboard
   jury
   hierarchy
   adversarial
   morph
