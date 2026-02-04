from mpi4agents.ring import RingAgent
from mpi4py import MPI
from argparse import ArgumentParser

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    agent = RingAgent(RANK, comm)
    agent.run(args.prompt)


if __name__ == "__main__":
    main()
