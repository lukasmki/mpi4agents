from argparse import ArgumentParser
from rich.console import Console
from rich.markdown import Markdown

from mpi4py import MPI

from _common import MODEL
from mpi4agents.hierarchy import HierarchyAgent

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--period", type=int, default=2)
    args = parser.parse_args()

    agent = HierarchyAgent(comm, MODEL, steps=args.steps, period=args.period)
    result = agent.run(args.prompt)

    console = Console()
    console.rule(f"Rank {RANK}")
    console.print(Markdown(result))
    console.rule()


if __name__ == "__main__":
    main()
