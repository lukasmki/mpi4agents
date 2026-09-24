Quickstart
==========

Running the examples
--------------------

The :file:`examples/` directory has one script per agent, numbered in the same order as
:doc:`agents/index`. Launch them with ``mpirun``; ``-n`` sets the number of agents:

.. code-block:: sh

   mpirun -n 4 uv run examples/3-tree.py --prompt "Why is the sky blue?"

Each rank prints its result under a ``Rank N`` heading. The ranks print independently, so
the headings can appear in any order.

Some agents take extra options, such as ``--rounds`` or ``--steps``. Run a script with
``--help`` to list them.

Choosing a model
----------------

All examples use the model defined in :file:`examples/_common.py`. It defaults to
pydantic-ai's ``TestModel``, which returns canned text without calling an LLM. That is
enough to check that the communication pattern runs to completion. To use a real model,
replace ``MODEL`` with any pydantic-ai model, for example an OpenAI-compatible local server:

.. code-block:: python

   from pydantic_ai.models.openai import OpenAIChatModel
   from pydantic_ai.providers.openai import OpenAIProvider

   MODEL = OpenAIChatModel(
       model_name="unsloth/Qwen3.5-4B-GGUF",
       provider=OpenAIProvider(base_url="http://127.0.0.1:8080"),
   )

Every rank makes its own LLM calls, so ``mpirun -n 8`` can send up to 8 requests at once.
Make sure the model server can handle that many concurrent requests. The page for each
agent lists how many LLM calls a rank makes.

Writing your own script
-----------------------

Every rank must construct the same agent and call ``run`` with the same prompt:

.. code-block:: python

   from mpi4py import MPI

   from mpi4agents.jury import JuryAgent

   agent = JuryAgent(MPI.COMM_WORLD, model)
   result = agent.run("Which sorting algorithm should I use for nearly sorted data?")
   print(f"Rank {agent.rank}: {result}")

What ``run`` returns depends on the rank and the pattern. For example, a
:class:`~mpi4agents.jury.JuryAgent` returns the verdict on rank 0 and each juror's vote on
the other ranks. Each agent's page describes its return value.
