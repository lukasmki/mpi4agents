from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent, MPIMessage, Tag


class FarmAgent(LLMAgent):
    """Manager-worker task farm: rank 0 splits the prompt into subtasks and hands them
    out on demand, workers solve one subtask at a time until told to stop"""

    def __init__(self, comm: MPI.Comm, model: Model):
        super().__init__(comm=comm, model=model)
        if self.size < 2:
            raise ValueError(
                "Task farm requires at least 2 ranks (1 manager, 1+ workers)"
            )

    def plan(self, prompt: str) -> list[str]:
        items = self.ask_list(
            prompt,
            "Break the PROMPT into small, independent subtasks. "
            "Reply with a list of subtasks and nothing else.",
        )
        return items or [prompt]

    def solve(self, prompt: str, task: str) -> str:
        return self.ask(
            f"OVERALL PROMPT:\n\n\t{prompt}\n\nSUBTASK:\n\n\t{task}",
            "Complete only the SUBTASK. Be concise.",
        )

    def synthesize(self, prompt: str, tasks: list[str], results: list[str]) -> str:
        results_str = "\n\n".join(
            f"SUBTASK: {t}\n\nRESULT:\n\n{r}" for t, r in zip(tasks, results)
        )
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\n{results_str}",
            "Combine the subtask RESULTs into a complete answer to the PROMPT.",
        )

    def manage(self, prompt: str) -> str:
        tasks = self.plan(prompt)
        results: list[str] = [""] * len(tasks)
        next_task = 0

        def dispatch(worker: int) -> bool:
            nonlocal next_task
            if next_task < len(tasks):
                msg = MPIMessage(self.rank, "TASK", (next_task, tasks[next_task]))
                self.send(dest=worker, msg=msg)
                next_task += 1
                return True
            self.send(dest=worker, msg=MPIMessage(self.rank, "STOP", None), tag=Tag.SYS)
            return False

        busy = sum(dispatch(worker) for worker in range(1, self.size))
        while busy:
            msg = self.recv(tag=Tag.MSG)
            index, result = msg.payload
            results[index] = result
            busy -= 1
            busy += dispatch(msg.sender)

        return self.synthesize(prompt, tasks, results)

    def work(self, prompt: str) -> str:
        log = []
        while True:
            msg = self.recv(source=0)
            if msg.kind == "STOP":
                break
            index, task = msg.payload
            result = self.solve(prompt, task)
            self.send(dest=0, msg=MPIMessage(self.rank, "RESULT", (index, result)))
            log.append(f"SUBTASK {index}: {task}\n\n{result}")
        return "\n\n".join(log)

    def run(self, prompt: str) -> str:
        return self.manage(prompt) if self.rank == 0 else self.work(prompt)
