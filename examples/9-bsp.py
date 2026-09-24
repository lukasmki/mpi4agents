from argparse import ArgumentParser
from rich.console import Console
from rich.markdown import Markdown

from mpi4py import MPI

from _common import MODEL
from mpi4agents.bsp import BSPAgent

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-supersteps", type=int, default=3)
    args = parser.parse_args()

    agent = BSPAgent(comm, MODEL, max_supersteps=args.max_supersteps)
    result = agent.run(args.prompt)

    console = Console()
    console.rule(f"Rank {RANK}")
    console.print(Markdown(result))
    console.rule()


if __name__ == "__main__":
    main()
