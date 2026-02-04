# mpi4agents

A proof of concept implementation of multi-agent systems parallelized over MPI.

## Multi-agent Systems (MAS) from communication patterns

Some standard communication patterns:

1. Ring
2. Pipeline
3. Tree
4. [Butterfly](https://en.wikipedia.org/wiki/Butterfly_network)
5. Manager-Worker or Task Farm
6. Nearest neighbor or Halo/Stencil
7. Token Passing (State passing)
8. Personalized All-to-All
9. Epoch/Bulk-Synchronous

Some other communication patterns:

1. Lattice/Grid
2. Blackboard/Shared-state
3. Voting/Jury
4. Intermittent hierarchy
5. Adversarial/Co-evolutionary
6. Role-morphing

### Manager-Worker

1. The manager agent decomposes a complex task into many substasks handled by the worker agents.
2. Each worker agent lives in a particular rank and retains its multi-turn message history.
3. Specialized worker agents live in a subset of ranks and may be called by worker agents.
4. At the end of a run, the manager agent compiles the worker agent results into a final result.
