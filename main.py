"""Experiments of the paper (Section 6), as independent tasks.

  python main.py plan [--quick] [--only ...]             write tasks/compute.txt and tasks/timing.txt
  python main.py task --list tasks/compute.txt --index 7  run one task (in an array: index from Slurm)
  python main.py gather                                   merge, write generated/ and figures_paper/
  python main.py all [--quick]                            plan, run every task here, gather
  python main.py warmup                                   compile the kernels once, before the arrays
  python main.py download                                 fetch the large networks (needs internet)

On a Slurm cluster, slurm/submit.sh plans, warms up, and submits the two arrays and the gather.
Every task resumes: its output is kept unless --force.
"""
import argparse
import os
import sys
import time

import numba

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments"))
import tasks as T   # noqa: E402


def split_names(text):
    """Split a comma-separated list of graph names, keeping commas inside parentheses ("BA(800, 2)")."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
            continue
        depth += (ch == "(") - (ch == ")")
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _plan_args(p):
    p.add_argument("--quick", action="store_true", help="small graphs and sizes")
    p.add_argument("--only", default=None, help="comma-separated graph names or tags of the registry")
    p.add_argument("--range-chunk-edges", type=int, default=500, help="edges per full-range task")
    p.add_argument("--range-max-edges", type=int, default=100_000, help="no full range above this size (0: none)")
    p.add_argument("--measures-chunk-edges", type=int, default=500)
    p.add_argument("--measures-max-edges", type=int, default=10_000)
    p.add_argument("--sizes", default=None, help="comma-separated sizes of the runtime benchmark")


def _task_args(p):
    p.add_argument("--threads", type=int, default=None, help="threads (default: Slurm's cores, else numba's)")
    p.add_argument("--reps", type=int, default=5, help="timing repetitions")
    p.add_argument("--milp-time-limit", type=float, default=60.0, help="seconds per integer program")
    p.add_argument("--range-budget", type=float, default=7200.0,
                   help="seconds of integer programs per task; edges beyond are left unsolved")
    p.add_argument("--force", action="store_true", help="recompute outputs that exist")


def _threads(args):
    th = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or numba.get_num_threads()
    numba.set_num_threads(min(th, numba.config.NUMBA_NUM_THREADS))
    return th


def _plan(args):
    nc, nt = T.plan(args.quick, split_names(args.only) if args.only else None, args.range_chunk_edges,
                    args.range_max_edges or None, args.measures_chunk_edges, args.measures_max_edges,
                    [int(x) for x in args.sizes.split(",")] if args.sizes else None)
    log(f"plan: {nc} compute tasks (tasks/compute.txt), {nt} timing tasks (tasks/timing.txt)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _plan_args(sub.add_parser("plan", help="write the task lists"))
    pt = sub.add_parser("task", help="run one task of a list")
    pt.add_argument("--list", required=True); pt.add_argument("--index", type=int, default=None)
    _task_args(pt)
    sub.add_parser("gather", help="merge the chunks and write the paper files")
    pa = sub.add_parser("all", help="plan, run every task in this process, gather")
    _plan_args(pa); _task_args(pa)
    sub.add_parser("warmup", help="compile the kernels")
    pd = sub.add_parser("download", help="fetch the large networks")
    pd.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.cmd == "plan":
        _plan(args)
    elif args.cmd == "task":
        idx = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        th = _threads(args)
        t = T.read_list(args.list)[idx]
        ran = T.run(args.list, idx, th, args.reps, args.milp_time_limit, args.range_budget, args.force)
        log(f"task {idx} {' '.join(t)}: {'done' if ran else 'already done'}")
    elif args.cmd == "gather":
        keys, rows, figs, problems = T.gather()
        log(f"gather: {len(keys)} numbers, {len(rows)} tables, {len(figs)} figures -> generated/, figures_paper/")
    elif args.cmd == "all":
        _plan(args)
        th = _threads(args)
        for lst in ("compute", "timing"):
            path = os.path.join(T.TASKS, f"{lst}.txt")
            for idx, t in enumerate(T.read_list(path)):
                ran = T.run(path, idx, th, args.reps, args.milp_time_limit, args.range_budget, args.force)
                log(f"{lst} {idx} {' '.join(t)}: {'done' if ran else 'already done'}")
        keys, rows, figs, problems = T.gather()
        log(f"gather: {len(keys)} numbers, {len(rows)} tables, {len(figs)} figures -> generated/, figures_paper/")
    elif args.cmd == "warmup":
        T.warmup(); log("warmup: kernels compiled")
    elif args.cmd == "download":
        T.download(args.quick)


if __name__ == "__main__":
    main()
