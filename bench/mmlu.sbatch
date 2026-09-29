#!/bin/bash
#SBATCH -A m2834_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J mas-bench
#SBATCH -o bench/results/%x-%j.out
# Serves the model with vLLM on this node's 4 GPUs and runs the harness against it with srun.
# Submit from the repository root; arguments are passed to harness.py:
#   sbatch bench/mmlu.sbatch --limit 500
#   RANKS=16 sbatch bench/mmlu.sbatch --agents ring tree --group-size 4
# The harness ranks only make HTTP requests, so they share the node with the server.
set -u
cd $SLURM_SUBMIT_DIR
RANKS=${RANKS:-4}
PORT=${PORT:-8000}
MODEL=google/gemma-4-12B-it
LOG=bench/results/$SLURM_JOB_NAME-$SLURM_JOB_ID.server.log
mkdir -p bench/results

export HF_HUB_OFFLINE=1
setsid $SCRATCH/vllm/start_vllm.sh --port $PORT > $LOG 2>&1 &
SPID=$!
stop_server() {
  kill -INT -- -$SPID 2>/dev/null
  for i in $(seq 60); do kill -0 -- -$SPID 2>/dev/null || return; sleep 1; done
  kill -9 -- -$SPID 2>/dev/null
}
trap stop_server EXIT

for i in $(seq 600); do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $LOG; exit 1; }
  sleep 2
done
curl -sf localhost:$PORT/health >/dev/null || { echo "server not ready after 20 min"; exit 1; }
echo "server ready after $(( i * 2 ))s"

# --no-sync keeps the mpi4py built against Cray MPICH (see README) and stops the ranks from
# all syncing the environment at once.
srun --overlap -n $RANKS uv run --no-sync bench/harness.py \
  --model $MODEL --base-url http://127.0.0.1:$PORT/v1 "$@"
