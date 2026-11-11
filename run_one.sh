#!/usr/bin/env bash
# Run the full nested-CV benchmark (CE, SupCon, CoCo, Dot-Regression) on ONE OpenML task.
#
#   ./run_one.sh 31                        # task 31 (credit-g), results in ./results
#   ./run_one.sh 31 --out res2 --threads 4
#
# Everything after the task id is passed to run_dataset.py. The log goes to <out>/logs/.
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: run_one.sh TASK_ID [run_dataset.py options...]"
    exit 1
fi
tid=$1; shift
cd "$(dirname "$(realpath "$0")")"

out=results
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [ "${args[$i]}" = "--out" ] && [ $((i + 1)) -lt ${#args[@]} ]; then out=${args[$((i + 1))]}; fi
done
mkdir -p "$out/logs"

# avoid BLAS thread over-subscription; torch threads are set with --threads
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

log="$out/logs/task$tid.log"
echo "$(date '+%F %T') start task $tid" | tee -a "$log"
python run_dataset.py --task-id "$tid" "$@" 2>&1 | tee -a "$log"
st=${PIPESTATUS[0]}
echo "$(date '+%F %T') end task $tid (exit $st)" | tee -a "$log"
exit "$st"
