from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

MSG_TAG = 1  # general message tag
CTRL_TAG = 2  # control/command messages


class MPIMessage:
    """Simple struct for typed messages."""

    def __init__(self, sender, kind, payload):
        self.sender = sender
        self.kind = kind
        self.payload = payload


class Agent:
    def __init__(self, rank: int, comm):
        self.rank = rank
        self.comm = comm

    def send(self, dest: int, msg: MPIMessage, tag: int = MSG_TAG):
        """Send object to dest."""
        self.comm.send(msg, dest=dest, tag=tag)

    def recv(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG):
        """Receive next message (blocking)."""
        return self.comm.recv(source=source, tag=tag)

    def probe(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG):
        """Check if a message is waiting."""
        return self.comm.iprobe(source=source, tag=tag)

    def run(self):
        """Override in subclass with main event/agent loop."""
        raise NotImplementedError


class MasterAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        task_queue = [f"task_{i}" for i in range(1, 10)]
        next_task = 0
        worker_count = SIZE - 1

        # Send initial tasks
        for worker_rank in range(1, SIZE):
            if next_task < len(task_queue):
                task = task_queue[next_task]
                self.send(worker_rank, MPIMessage(self.rank, "TASK", task))
                print(f"[Master] sent {task} to worker {worker_rank}")
                next_task += 1
            else:
                self.send(worker_rank, MPIMessage(self.rank, "STOP", None))

        # Collect and reassign until done
        finished = 0
        while finished < len(task_queue):
            msg = self.recv(tag=MSG_TAG)
            if msg.kind == "RESULT":
                print(f"[Master] got {msg.payload} from worker {msg.sender}")
                finished += 1

            if next_task < len(task_queue):
                task = task_queue[next_task]
                self.send(msg.sender, MPIMessage(self.rank, "TASK", task))
                print(f"[Master] reassigned {task} to worker {msg.sender}")
                next_task += 1
            else:
                self.send(msg.sender, MPIMessage(self.rank, "STOP", None))

        print("[Master] All tasks processed.")


class WorkerAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_task(self, payload):
        """Example task handling."""
        time.sleep(0.2)
        return f"result_{payload}_by_{self.rank}"

    def run(self):
        while True:
            msg = self.recv(source=0, tag=MPI.ANY_TAG)
            if msg.kind == "TASK":
                if msg.payload is None:
                    # stop signal
                    break
                result = self.handle_task(msg.payload)
                self.send(0, MPIMessage(self.rank, "RESULT", result))
            elif msg.kind == "STOP":
                break
            else:
                print(f"[Worker {self.rank}] unknown msg type: {msg.kind}")


# class EventDrivenAgent(Agent):
#     handlers = {}

#     def on(self, kind):
#         def decorator(fn):
#             self.handlers[kind] = fn
#             return fn
#         return decorator

#     def run(self):
#         while True:
#             msg = self.recv()
#             handler = self.handlers.get(msg.kind)
#             if handler:
#                 handler(msg)

# class MyWorker(EventDrivenAgent):
#     @EventDrivenAgent.on("TASK")
#     def handle_task(self, msg):
#         time.sleep(0.2)
#         return f"result_{msg.payload}_by_{self.rank}"


def main():
    if RANK == 0:
        master = MasterAgent(RANK, comm)
        master.run()
    else:
        worker = WorkerAgent(RANK, comm)
        worker.run()


if __name__ == "__main__":
    main()
