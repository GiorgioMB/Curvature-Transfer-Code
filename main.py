"""Experiments of the paper (Section 6). Every step resumes from results/ unless --force.

  python main.py                      every step, as in the paper
  python main.py --quick              a smoke run of every step on small graphs and sizes
  python main.py --steps tightness,range --only "BA(800, 2),Cora"
  python main.py --steps paper        only rebuild generated/numbers.tex, INDEX.md and the figures
"""
import argparse
import json
import os
import sys
import time

import numba

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments"))
import large_graphs          # noqa: E402
import suite                 # noqa: E402
import tq_artifacts as A     # noqa: E402
import tq_experiments as X   # noqa: E402

STEPS = ["tightness", "range", "measures", "kstar", "screening", "benchmark", "paper"]
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
SIZES_QUICK = [1_000, 3_000, 10_000]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    ap.add_argument("--quick", action="store_true", help="small graphs and sizes, one repetition")
    ap.add_argument("--only", default=None, help="comma-separated graph names or tags of the registry")
    ap.add_argument("--threads", type=int, default=None, help="threads (default: all of numba's)")
    ap.add_argument("--reps", type=int, default=None, help="timing repetitions (default 5; 1 with --quick)")
    ap.add_argument("--force", action="store_true", help="recompute results that exist")
    ap.add_argument("--range-max-edges", type=int, default=None, help="skip the full range above this size")
    ap.add_argument("--measures-max-edges", type=int, default=10_000)
    ap.add_argument("--milp-time-limit", type=float, default=60.0, help="seconds per integer program")
    ap.add_argument("--range-budget", type=float, default=7200.0,
                    help="seconds of integer programs per graph; edges beyond are left unsolved")
    ap.add_argument("--sizes", default=None, help="comma-separated sizes of the runtime benchmark")
    args = ap.parse_args()
    threads = args.threads or numba.get_num_threads()
    numba.set_num_threads(threads)
    reps = args.reps or (1 if args.quick else 5)
    os.makedirs(X.RESULTS, exist_ok=True)
    with open(os.path.join(X.RESULTS, "env.json"), "w") as fh:
        json.dump(X.environment(threads, reps), fh)
    only = split_names(args.only) if args.only else None
    reg = suite.specs(only=only, quick=args.quick)
    table = [s for s in reg if s.group in ("random", "real")]
    steps = [s.strip() for s in args.steps.split(",")]
    for s in steps:
        if s not in STEPS:
            raise SystemExit(f"unknown step {s}; steps: {STEPS}")
    if "tightness" in steps:
        for s in reg:
            r = X.tightness(s, threads, args.force)
            log(f"tightness {s.name}: |E|={r['m']} TQ {r['tq_sign']:.1f}% prior {r['prior_sign']:.1f}% "
                f"(violations {r['tq_violations']}/{r['prior_violations']})")
    if "range" in steps:
        cap = args.range_max_edges or (5_000 if args.quick else None)
        for s in table:
            if cap and X.load_json("tightness", s.tag)["m"] > cap:
                continue
            r = X.full_range(s, threads, args.milp_time_limit, args.force, args.range_budget)
            log(f"range {s.name}: TQ {r['tq_sign']:.1f}% full range {r['range_sign']:.1f}% gaps "
                f"{r['gap_lo']:.4f}/{r['gap_hi']:.4f} complete={r['complete']} ({r['seconds']:.0f}s)")
    if "measures" in steps:
        for s in table:
            if X.load_json("tightness", s.tag)["m"] > args.measures_max_edges:
                continue
            r = X.measures(s, threads, args.milp_time_limit, args.force, args.range_budget)
            log(f"measures {s.name}: " + "  ".join(
                f"{k} {r[k]['tq_sign']:.1f}/{r[k]['range_sign']:.1f}%" for k in ("nonlazy", "half", "degree")))
    if "kstar" in steps:
        for s in table:
            n, E = suite.load(s)
            r = X.kstar(s.tag, s.name, n, E, threads, reps, args.force)
            if r.get("undecided"):
                log(f"kstar {s.name}: {r['undecided']} undecided, middle stage decides "
                    f"{r.get('decided_by_kstar', 0):.1f}%, slowdown {r.get('slowdown', 0):.2f}x")
    if "screening" in steps:
        nets = [(ls.tag, ls.name, ls) for ls in large_graphs.specs(quick=args.quick)]
        if not args.quick:
            nets += [(s.tag, s.name, s) for s in suite.specs(only=["PubMed", "Arxiv"])]
        for tag, name, sp in nets:
            if hasattr(sp, "url"):
                n, E, _ = large_graphs.load(sp)
            else:
                n, E = suite.load(sp)
            r = X.screening(tag, name, n, E, threads, reps, args.force)
            log(f"screening {name}: |E|={r['m']} full {r['t_full']:.2f}s TQ {r['t_tq']:.2f}s decided "
                f"{r['decided']:.1f}% speedup {r['speedup']:.1f}x exact={r['exact_answer']}")
    if "benchmark" in steps:
        sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else (SIZES_QUICK if args.quick else SIZES)
        for fam in X.FAMILIES:
            b = X.benchmark(fam, sizes, threads, reps, args.force)
            log(f"benchmark {fam}: " + ", ".join(f"|E|={p['m']}: {p['t_full']:.2f}/{p['t_tq']:.3f}/{p['t_bf']:.3f}s"
                                                 for p in b["points"]))
    if "paper" in steps:
        keys, rows, figs, problems = A.write()
        log(f"paper: {len(keys)} numbers, {len(rows)} tables, {len(figs)} figures -> generated/, figures_paper/")


if __name__ == "__main__":
    main()
