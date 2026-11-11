#!/usr/bin/env bash
# Run every OpenML-CC18 task (each dataset in its own process), then the statistical analysis.
#
#   ./run_all.sh                    # 1 dataset at a time, all CPU threads
#   ./run_all.sh -j 6               # 6 datasets in parallel, threads split between them
#   ./run_all.sh -j 4 -o res -- --device auto    # arguments after -- go to run_dataset.py
#   ./run_all.sh --only 31 37 54    # only these task ids
#
# Finished datasets (<out>/task*/DONE) are skipped; unfinished ones resume at the
# first missing (method, outer fold). Logs: <out>/logs/task<ID>.log
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

jobs_n=1; out=results; list=cc18_tasks.tsv; threads=""; only=(); no_analysis=0; extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        -j|--jobs) jobs_n=$2; shift 2 ;;
        -o|--out) out=$2; shift 2 ;;
        -l|--list) list=$2; shift 2 ;;
        -t|--threads) threads=$2; shift 2 ;;
        --only) shift; while [ $# -gt 0 ] && [[ "$1" != -* ]]; do only+=("$1"); shift; done ;;
        --no-analysis) no_analysis=1; shift ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

ncpu=$(getconf _NPROCESSORS_ONLN)
if [ -z "$threads" ]; then
    threads=$((ncpu / jobs_n)); [ "$threads" -lt 1 ] && threads=1
fi
echo "CPUs: $ncpu  parallel datasets: $jobs_n  torch threads per dataset: $threads"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$out/logs"

if [ ${#only[@]} -gt 0 ]; then
    tids=("${only[@]}")
else
    if [ ! -f "$list" ]; then
        echo "downloading CC18 task list -> $list"
        python list_cc18.py --out "$list" || exit 1
    fi
    mapfile -t tids < <(tail -n +2 "$list" | cut -f1)
fi
echo "tasks to run: ${#tids[@]}"

is_done() { compgen -G "$out/task$1_*/DONE" > /dev/null; }

for tid in "${tids[@]}"; do
    if is_done "$tid"; then echo "task $tid already done, skipping"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$jobs_n" ]; do sleep 5; done
    echo "$(date +%T) launch task $tid"
    python run_dataset.py --task-id "$tid" --out "$out" --threads "$threads" "${extra[@]}" \
        >> "$out/logs/task$tid.log" 2>&1 &
done
wait
echo "$(date +%T) all datasets finished"

for tid in "${tids[@]}"; do
    is_done "$tid" || echo "WARNING: task $tid incomplete, see $out/logs/task$tid.log"
done

if [ "$no_analysis" -eq 0 ]; then
    python analysis/analyze_results.py --results "$out" --out "$out/analysis"
fi
