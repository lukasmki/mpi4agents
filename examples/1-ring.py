from argparse import ArgumentParser

from mpi4py import MPI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from mpi4agents.ring import RingAgent

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()
model = OpenAIChatModel(
    model_name="unsloth/Qwen3.5-4B-GGUF",
    provider=OpenAIProvider(base_url="http://127.0.0.1:8080"),
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    agent = RingAgent(comm, model)
    agent.run(args.prompt)


if __name__ == "__main__":
    main()
