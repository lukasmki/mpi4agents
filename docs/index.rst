mpi4agents
==========

A proof of concept implementation of multi-agent systems parallelized over MPI.

Each MPI rank runs one LLM agent. The agents coordinate using classic parallel
communication patterns, such as rings, trees, butterflies and task farms, so that a
multi-agent system's structure is simply its communication pattern.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   concepts
   agents/index
   api

Installation
------------

mpi4agents needs Python 3.13 or later and an MPI implementation, such as Open MPI or
MPICH, installed on the system.

.. code-block:: sh

   uv sync

Quick example
-------------

.. code-block:: python

   from mpi4py import MPI
   from pydantic_ai.models.test import TestModel

   from mpi4agents.tree import TreeAgent

   agent = TreeAgent(MPI.COMM_WORLD, TestModel())
   result = agent.run("Why is the sky blue?")

.. code-block:: sh

   mpirun -n 4 uv run my_script.py
