# mpi4agents

A proof of concept implementation of multi-agent systems parallelized over MPI.

Docs: http://lukasmki.github.io/mpi4agents/

## Multi-agent Systems (MAS) from communication patterns

Some standard communication patterns:

| Pattern | Agent | Example | Ranks |
| --- | --- | --- | --- |
| Ring | [`RingAgent`](src/mpi4agents/ring.py) | [1-ring.py](examples/1-ring.py) | any |
| Pipeline | [`PipeAgent`](src/mpi4agents/pipe.py) | [2-pipe.py](examples/2-pipe.py) | any |
| Tree | [`TreeAgent`](src/mpi4agents/tree.py) | [3-tree.py](examples/3-tree.py) | any |
| [Butterfly](https://en.wikipedia.org/wiki/Butterfly_network) | [`ButterflyAgent`](src/mpi4agents/butterfly.py) | [4-butterfly.py](examples/4-butterfly.py) | power of 2 |
| Manager-Worker or Task Farm | [`FarmAgent`](src/mpi4agents/farm.py) | [5-farm.py](examples/5-farm.py) | ≥ 2 |
| Nearest neighbor or Halo/Stencil | [`HaloAgent`](src/mpi4agents/halo.py) | [6-halo.py](examples/6-halo.py) | any |
| Token Passing (State passing) | [`TokenAgent`](src/mpi4agents/token_passing.py) | [7-token.py](examples/7-token.py) | any |
| Personalized All-to-All | [`AllToAllAgent`](src/mpi4agents/alltoall.py) | [8-alltoall.py](examples/8-alltoall.py) | any |
| Epoch/Bulk-Synchronous | [`BSPAgent`](src/mpi4agents/bsp.py) | [9-bsp.py](examples/9-bsp.py) | any |

Some other communication patterns:

| Pattern | Agent | Example | Ranks |
| --- | --- | --- | --- |
| Lattice/Grid | [`GridAgent`](src/mpi4agents/grid.py) | [10-grid.py](examples/10-grid.py) | any |
| Blackboard/Shared-state | [`BlackboardAgent`](src/mpi4agents/blackboard.py) | [11-blackboard.py](examples/11-blackboard.py) | ≥ 2 |
| Voting/Jury | [`JuryAgent`](src/mpi4agents/jury.py) | [12-jury.py](examples/12-jury.py) | any |
| Intermittent hierarchy | [`HierarchyAgent`](src/mpi4agents/hierarchy.py) | [13-hierarchy.py](examples/13-hierarchy.py) | any |
| Adversarial/Co-evolutionary | [`AdversarialAgent`](src/mpi4agents/adversarial.py) | [14-adversarial.py](examples/14-adversarial.py) | even |
| Role-morphing | [`MorphAgent`](src/mpi4agents/morph.py) | [15-morph.py](examples/15-morph.py) | any |

## Running the examples

The examples share the model defined in [examples/_common.py](examples/_common.py)
(a `TestModel` by default; swap in a real model there).

```sh
mpirun -n 4 uv run examples/3-tree.py --prompt "Why is the sky blue?"
```
