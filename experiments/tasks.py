"""The experiments as independent tasks, for Slurm job arrays.

  plan      write tasks/compute.txt (throughput tasks) and tasks/timing.txt (tasks whose times are reported,
            for dedicated nodes); one task per line, tab-separated
  run       run line ``index`` of a task list, and record its provenance in results/_tasks/
  gather    merge the chunks and write generated/numbers.tex, generated/INDEX.md and the figures
  warmup    compile every kernel once, so that the tasks share the numba cache instead of racing to write it
  download  fetch the large networks into experiments/data/large/ (for clusters whose nodes are offline)

The provenance of a task (wall and CPU time, allocated cores, peak memory, node, processor, cluster, Slurm
ids, software versions) feeds the keys of the code availability statement.
"""
import json
import math
import os
import platform
import resource
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import large_graphs          # noqa: E402
import suite                 # noqa: E402
import tq_experiments as X   # noqa: E402

TASKS = os.path.join(ROOT, "tasks")
PROV = os.path.join(X.RESULTS, "_tasks")
TIMING_STEPS = ("kstar", "screening", "benchmark")
BENCH_SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
BENCH_SIZES_QUICK = [1_000, 3_000, 10_000]


# -------------------------------------------------------------------------------------------------- plan
def _screening_networks(quick):
    nets = [("large", ls.tag) for ls in large_graphs.specs(quick=quick)]
    if not quick:
        nets += [("registry", s.tag) for s in suite.specs(only=["PubMed", "Arxiv"])]
    return nets


def plan(quick=False, only=None, range_chunk_edges=500, range_max_edges=100_000, measures_chunk_edges=500,
         measures_max_edges=10_000, sizes=None):
    """Write the task lists; returns (number of compute tasks, number of timing tasks)."""
    table = [s for s in suite.specs(only=only, quick=quick) if s.group in ("random", "real")]
    compute, timing = [], []
    for s in table:
        compute.append(("tightness", s.tag))
        m = len(suite.load(s)[1])
        if not range_max_edges or m <= range_max_edges:
            K = max(1, math.ceil(m / range_chunk_edges))
            compute += [("range", s.tag, str(c), str(K)) for c in range(K)]
        if m <= measures_max_edges:
            K = max(1, math.ceil(m / measures_chunk_edges))
            for name, _, _ in X.MEASURES:
                compute += [("measures", s.tag, name, str(c), str(K)) for c in range(K)]
        timing.append(("kstar", s.tag))
    timing += [("screening", src, tag) for src, tag in _screening_networks(quick)]
    for fam in X.FAMILIES:
        timing += [("benchmark", fam, str(n)) for n in (sizes or (BENCH_SIZES_QUICK if quick else BENCH_SIZES))]
    os.makedirs(TASKS, exist_ok=True)
    for name, lst in (("compute", compute), ("timing", timing)):
        with open(os.path.join(TASKS, f"{name}.txt"), "w") as fh:
            fh.write("".join("\t".join(t) + "\n" for t in lst))
    with open(os.path.join(TASKS, "plan.json"), "w") as fh:
        json.dump({"quick": quick, "only": only, "range_chunk_edges": range_chunk_edges,
                   "range_max_edges": range_max_edges, "measures_chunk_edges": measures_chunk_edges,
                   "measures_max_edges": measures_max_edges, "sizes": sizes}, fh, indent=1)
    return len(compute), len(timing)


def read_list(path):
    with open(path) as fh:
        return [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]


# ----------------------------------------------------------------------------------------------- one task
def _spec(tag):
    return suite.specs(only=[tag])[0]


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.lower().startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _versions():
    import numba
    import orcut
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "orcut": orcut.__version__}


def _dispatch(t, threads, reps, limit, budget, force):
    step = t[0]
    if step == "tightness":
        before = X.done("tightness", t[1])
        X.tightness(_spec(t[1]), threads, force)
        return force or not before
    if step == "range":
        return X.range_chunk(_spec(t[1]), int(t[2]), int(t[3]), threads, limit, budget, force)
    if step == "measures":
        return X.measures_chunk(_spec(t[1]), t[2], int(t[3]), int(t[4]), threads, limit, budget, force)
    if step == "kstar":
        before = X.done("kstar", t[1])
        sp = _spec(t[1]); n, E = suite.load(sp)
        X.kstar(sp.tag, sp.name, n, E, threads, reps, force)
        return force or not before
    if step == "screening":
        before = X.done("screening", t[2])
        if t[1] == "large":
            ls = [x for x in large_graphs.specs() if x.tag == t[2]][0]
            n, E, _ = large_graphs.load(ls); name = ls.name
        else:
            sp = _spec(t[2]); n, E = suite.load(sp); name = sp.name
        X.screening(t[2], name, n, E, threads, reps, force)
        return force or not before
    if step == "benchmark":
        return X.benchmark_point(t[1], int(t[2]), threads, reps, force)
    raise ValueError(f"unknown task {t}")


def run(task_list, index, threads, reps=5, limit=60.0, budget=7200.0, force=False):
    """Run one task; returns True if it ran (False if its output existed)."""
    t = read_list(task_list)[index]
    t0 = time.time(); r0 = resource.getrusage(resource.RUSAGE_SELF)
    ran = _dispatch(t, threads, reps, limit, budget, force)
    t1 = time.time(); r1 = resource.getrusage(resource.RUSAGE_SELF); rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    if ran:
        cores = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or threads)
        rec = {"task": t, "list": os.path.basename(task_list), "start": t0, "end": t1, "wall": t1 - t0,
               "cpu": (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime) + rc.ru_utime + rc.ru_stime,
               "cores": cores, "threads": int(threads), "reps": int(reps),
               "maxrss_gb": max(r1.ru_maxrss, rc.ru_maxrss) / 1e6,     # Linux: kilobytes
               "host": socket.gethostname(), "cpu_model": _cpu_model(),
               "cluster": os.environ.get("SLURM_CLUSTER_NAME", ""),
               "slurm": {k: os.environ.get(k, "") for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                                                            "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION")},
               "versions": _versions()}
        os.makedirs(PROV, exist_ok=True)
        with open(os.path.join(PROV, "-".join(t).replace("/", "_") + ".json"), "w") as fh:
            json.dump(rec, fh, indent=1)
    return ran


# ------------------------------------------------------------------------------------------------- gather
def gather():
    import tq_artifacts as A
    for s in suite.REGISTRY:
        X.merge_range(s)
        X.merge_measures(s)
    for fam in X.FAMILIES:
        X.merge_benchmark(fam)
    return A.write()


# ------------------------------------------------------------------------------------- warmup and download
def warmup():
    """Compile every kernel on a small graph (fills the numba cache before the arrays start)."""
    import networkx as nx
    from tqbound import bf, bm_tlw, lower_and_relaxed, screen_then_solve, tq_general, tq_lazy, upper_end
    G = nx.powerlaw_cluster_graph(60, 3, 0.5, seed=1)
    n, E = G.number_of_nodes(), np.array(G.edges(), np.int64)
    g, ge, ip, ix = X.build(n, E)
    tq_lazy(ip, ix, ge); bf(ip, ix, ge); bm_tlw(ip, ix, ge)
    o = np.empty((len(ge), 4))
    for mode in ((0, 0.0, 0), (1, 0.0, 0), (2, 0.0, 0), (0, 0.0, 1)):
        tq_general(ip, ix, ge[:, 0].copy(), ge[:, 1].copy(), *mode, o, 2)
    adj, deg = X.adjacency(ip, ix)
    edges = [(int(a), int(b)) for a, b in ge[:10]]
    lower_and_relaxed(adj, deg, edges, 0.0, exact=True)
    upper_end(adj, deg, 0.0, *edges[0])
    screen_then_solve(g, 0)


def download(quick=False):
    for ls in large_graphs.specs(quick=quick):
        n, E, _ = large_graphs.load(ls)
        print(f"{ls.name}: n={n} |E|={len(E)}", flush=True)
