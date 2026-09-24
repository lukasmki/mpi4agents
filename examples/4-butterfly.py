from argparse import ArgumentParser
from rich.console import Console
from rich.markdown import Markdown

from mpi4py import MPI

from _common import MODEL
from mpi4agents.butterfly import ButterflyAgent

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print intermediate answers"
    )
    args = parser.parse_args()

    agent = ButterflyAgent(comm, MODEL)
    console = Console()

    if args.verbose:
        for step, answer in enumerate(agent.irun(args.prompt)):
            console.rule(f"Rank {RANK} · step {step}")
            console.print(Markdown(answer))
        console.rule()
    else:
        result = agent.run(args.prompt)
        console.rule(f"Rank {RANK}")
        console.print(Markdown(result))
        console.rule()


if __name__ == "__main__":
    main()
