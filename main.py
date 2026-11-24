#!/usr/bin/env python3
"""Reproduce every table, figure, and number of Section 6 with one command.

    python main.py                       # everything
    python main.py --quick               # small smoke run of the whole pipeline (minutes)
    python main.py --steps bounds paper  # selected steps only
    python main.py --list                # the steps and what each one produces
    python main.py --steps paper --paper ~/overleaf --manuscript tex/manuscript-journal.tex

Outputs go to ``results/`` (``--out``), logs to ``logs/<step>.log`` (``--logs``). Every step saves
its work as soon as it is computed (per graph, per chunk of edges, per graph size), with atomic
writes, so a killed run resumes when the same command is run again: finished steps are skipped
through their stamp in ``results/.stamps`` (``--force`` reruns them, still reusing the saved
units; delete ``results/`` to start over). The ``paper`` step writes the files the manuscript reads
(``experiments/paper_artifacts.py``); ``consistency`` cross-checks tables, figures and benchmarks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")
PY = sys.executable
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)
from curvature_transfer import _io  # noqa: E402
import suite  # noqa: E402

# Version of the timed computations (benchmark, screening). Their stamps include
# it, so that a step finished with older timing code is rerun (2: the 2-hop
# kernel, which also evaluates the TLW certificate).
TIMING_VERSION = 2

# Figures that the "figures" step collects into results/figures/ (the names the manuscript
# includes), from the output directories of the steps that draw them (FIGURE_DIRS, under results/);
# the runtime figures of "benchmark" are collected separately.
_tag = lambda name: suite.BY_NAME[name].file_tag
FIGURES = ([f"fig_scatter_cBF_cOR__{_tag(g)}_2d_Hist.png" for g in suite.SCATTER_GRAPHS]
           + [f"fig_cOR_{v}__{_tag(g)}.png" for g in suite.DIST_GRAPHS.values() for v in ("hist", "quantile")]
           + [f"fig_screening_map__{_tag(suite.SCREENING_MAP_GRAPH)}.pdf"])
FIGURE_DIRS = [("edges", "csv", "figures_paper"), ("screening_map",)]

STEPS = {
    "environment": "software/hardware record -> environment.json",
    "tests": "pytest suite: exact OT vs LP, theoretical bounds on random graphs",
    "examples": "closed-form values (Q_n, K_n, trees, grid) -> tables/examples.txt",
    "edges": "per-edge statistics, bounds, exact curvatures of every registry graph (chunked) -> edges/",
    "bounds": "bound comparison and ranking tables, from edges/ -> tables/bounds/",
    "screening": "certify-then-solve evaluation, from edges/ -> tables/screening/",
    "screening_map": "certify-then-solve edge map (Section 6.3 figure) and its numbers, from edges/ -> screening_map/",
    "distributions": "per-edge CSVs + distribution figures, from edges/ -> edges/csv/",
    "benchmark": "runtime scaling, one figure per family -> benchmark/",
    "figures": "collect key figures into one directory -> figures/",
    "consistency": "cross-checks across bounds, screening, figures, and benchmarks -> CONSISTENCY.md",
    "paper": "numbers.tex, table rows, INDEX.md, figures for the manuscript -> --paper (generated/, figures_paper/)",
    "report": "REPORT.md mapping experimental artifacts to files",
}
CORE = [
    "environment",
    "tests",
    "examples",
    "edges",
    "bounds",
    "screening",
    "screening_map",
    "distributions",
    "benchmark",
    "figures",
]


# -------------------------------------------------------------- runner
class Runner:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        self.paper_dir = os.path.abspath(a.paper or os.path.join(self.out, "paper"))
        self.logs = os.path.abspath(a.logs)
        os.makedirs(self.logs, exist_ok=True)
        for d in (".stamps", "tables", "figures"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg")
        self.env["PYTHONPATH"] = (
            ROOT + os.pathsep + self.env.get("PYTHONPATH", "")
        )
        if a.jobs is not None:
            self.env["NUMBA_NUM_THREADS"] = str(
                os.cpu_count() if a.jobs == -1 else a.jobs
            )
        self.status = {}
        n = _io.cleanup_tmp(self.out)
        if n:
            print(
                f"[main] removed {n} leftover temporary file(s) from an interrupted run"
            )

    def path(self, *p):
        return os.path.join(self.out, *p)

    def stamp(self, step, cfg):
        h = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:16]
        return self.path(".stamps", f"{step}.{h}.done")

    def run(self, step, cmd, cfg=None, cwd=EXP):
        cfg = {"cmd": cmd, **(cfg or {})}
        st = self.stamp(step, cfg)
        if os.path.exists(st) and not self.a.force:
            print(f"[skip] {step} (already done; --force to redo)")
            self.status[step] = "skipped (done)"
            return True
        log = os.path.join(self.logs, f"{step}.log")
        print(f"[run ] {step}: {' '.join(cmd)}\n       log: {log}", flush=True)
        t0 = time.time()
        with open(log, "a") as fh:
            fh.write(
                f"\n===== {datetime.now(timezone.utc).isoformat()} {' '.join(cmd)}\n"
            )
            fh.flush()
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                if self.a.verbose:
                    sys.stdout.write("       " + line)
            p.wait()
        dt = time.time() - t0
        if p.returncode != 0:
            print(
                f"[FAIL] {step} after {dt:.0f}s (exit {p.returncode}); see {log}"
            )
            self.status[step] = f"FAILED (see {log})"
            return False
        _io.atomic_text(datetime.now(timezone.utc).isoformat(), st)
        print(f"[ ok ] {step} ({dt:.0f}s)")
        self.status[step] = f"ok ({dt:.0f}s)"
        return True

    # ------------------------------------------------------------- steps
    def environment(self):
        info = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "argv": sys.argv,
            "quick": self.a.quick,
            "seed": self.a.seed,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        for mod in (
            "numpy",
            "scipy",
            "numba",
            "networkx",
            "pandas",
            "matplotlib",
        ):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:
                info[mod] = None
        try:
            info["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            info["git_commit"] = None
        _io.atomic_json(info, self.path("environment.json"))
        self.status["environment"] = "ok"
        return True

    def common(self):
        c = [
            "--seed",
            str(self.a.seed),
            "--chunk",
            str(self.a.chunk),
        ] + (["--quick"] if self.a.quick else [])
        return c + ([] if self.a.jobs is None else ["--jobs", str(self.a.jobs)])

    def tests(self):
        return self.run(
            "tests",
            [PY, "-m", "pytest", "-q", os.path.join(ROOT, "tests")],
            cwd=ROOT,
        )

    def examples(self):
        return self.run(
            "examples",
            [
                PY,
                "check_examples.py",
                "--out",
                self.path("tables", "examples.txt"),
            ],
        )

    def edges(self):
        return self.run(
            "edges", [PY, "edges.py", "--out", self.out] + self.common()
        )

    def bounds(self):
        return self.run(
            "bounds",
            [
                PY,
                "compare_bounds.py",
                "--results",
                self.out,
                "--out",
                self.path("tables", "bounds"),
            ]
            + self.common(),
        )

    def screening(self):
        return self.run("screening", 
                        [PY, "screening.py", "--results", 
                             self.out, "--out", self.path("tables", "screening"),
                            "--data-dir", self.path("data", "large")] + self.common(),
                        cfg={"timing_version": TIMING_VERSION})

    def screening_map(self):
        return self.run(
            "screening_map",
            [
                PY,
                "screening_map.py",
                "--results",
                self.out,
                "--out",
                self.path("screening_map"),
                "--graph",
                suite.SCREENING_MAP_GRAPH,
            ]
            + self.common(),
        )

    def distributions(self):
        return self.run(
            "distributions",
            [PY, "distributions.py", "--results", self.out] + self.common(),
        )

    def benchmark(self):
        ok = True
        mx, ng = (
            (800, 6)
            if self.a.quick
            else (self.a.bench_max_n, self.a.bench_graphs)
        )
        for fam in self.a.bench_families:
            cmd = [
                PY,
                "benchmark.py",
                "--family",
                fam,
                "--min-n",
                "50",
                "--max-n",
                str(mx),
                "--graphs",
                str(ng),
                "--seed",
                "42",
                "--budget-hours",
                str(self.a.bench_budget_hours),
                "--out",
                self.path("benchmark"),
            ]
            cmd += [] if self.a.jobs is None else ["--jobs", str(self.a.jobs)]
            cmd += (
                []
                if self.a.bench_mem_gb is None
                else ["--mem-gb", str(self.a.bench_mem_gb)]
            )
            ok &= self.run(f"benchmark_{fam}", cmd, cfg={"timing_version": TIMING_VERSION})
        return ok

    def figures(self):
        dst = self.path("figures")
        srcs = [self.path(*d) for d in FIGURE_DIRS]
        missing = []
        for f in FIGURES:
            src = next((s for s in srcs if os.path.exists(os.path.join(s, f))), None)
            if src is not None:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            else:
                missing.append(f)
        for fam in self.a.bench_families:
            rt = self.path("benchmark", f"runtime_scaling_loglog_{fam.lower()}.pdf")
            if os.path.exists(rt):
                shutil.copy2(rt, os.path.join(dst, f"runtime_scaling_{fam.lower()}.pdf"))
            else:
                missing.append(f"runtime_scaling_{fam.lower()}.pdf")
        if missing:
            print(
                f"[warn] figures not produced: {missing}"
                + (" (expected with --quick)" if self.a.quick else "")
            )
        self.status["figures"] = (
            "ok" if not missing else f"ok, {len(missing)} missing"
        )
        return True

    def consistency(self):
        import paper_artifacts as PA

        checks = PA.consistency_checks(self.out)
        failed = [c for c in checks if not c[1]]
        lines = [
            "# Consistency checks",
            "",
            f"Generated {datetime.now(timezone.utc).isoformat()}",
            "",
        ]
        lines += [
            f"- [{'ok' if ok else 'FAIL'}] {name}"
            + (f" — {d}" if d and not ok else "")
            for name, ok, d in checks
        ]
        _io.atomic_text("\n".join(lines) + "\n", self.path("CONSISTENCY.md"))
        print(
            f"[{'ok' if not failed else 'FAIL'}] consistency: {len(checks) - len(failed)}/{len(checks)} checks passed"
            + "".join(f"\n       FAIL: {n} ({d})" for n, _, d in failed)
        )
        if failed:
            self.status["consistency"] = (
                f"FAILED ({len(failed)}/{len(checks)} checks failed; see CONSISTENCY.md)"
            )
            return False
        self.status["consistency"] = f"ok ({len(checks)}/{len(checks)} passed)"
        return True

    def paper(self):
        import paper_artifacts as PA

        info = PA.build(self.out, self.paper_dir, self.a.manuscript)
        miss = len(info["missing"]) + len(info["missing_rows"])
        print(f"[ ok ] paper: {info['numbers']} numbers, {len(info['tables'])} tables -> "
              f"{os.path.join(self.paper_dir, 'generated')}" + (f"; the manuscript misses {miss} (see INDEX.md)"
                                                          if miss else ""))
        self.status["paper"] = "ok" if not miss else f"ok, {miss} keys of the manuscript missing"
        return True

    def report(self):
        mtag = suite.BY_NAME[suite.SCREENING_MAP_GRAPH].file_tag
        rows = [
            ("Per-edge arrays behind every table and figure", "edges", "<tag>.paper.npz, <tag>.exact.npz"),
            ("Bound comparison, ranking, BF interval (Tables 3-6)", "tables/bounds/bound_comparison.csv", ""),
            ("Certify-then-solve on the registry graphs", "tables/screening/screening_benchmark.csv", ""),
            ("Certify-then-solve on large networks (Table 7, timed)", "tables/screening/screening_large.csv", ""),
            ("Edge map of certify-then-solve (Section 6.3)", f"figures/fig_screening_map__{mtag}.pdf",
             suite.SCREENING_MAP_GRAPH),
            ("Runtime scaling (Table 8, one figure per family)", "figures", "runtime_scaling_<family>.pdf"),
            ("Scatter and distributional views", "figures", "fig_*.png"),
            ("Closed-form values on structured graphs", "tables/examples.txt", ""),
            ("Cross-checks", "CONSISTENCY.md", ""),
            ("Files read by the manuscript", os.path.relpath(self.paper_dir, self.out), "generated/INDEX.md"),
        ]
        lines = ["# Reproduction report", "",
                 f"Generated {datetime.now(timezone.utc).isoformat()} by `{' '.join(sys.argv)}`"
                 + ("  \n**Quick mode: reduced graphs and sizes.**" if self.a.quick else ""), "",
                 "| Artifact | File (under results/) | Present | Notes |", "| --- | --- | --- | --- |"]
        lines += [f"| {n} | `{p}` | {'yes' if os.path.exists(self.path(p)) else '**missing**'} | {note} |"
                  for n, p, note in rows]
        lines += ["", "## Step status", ""] + [f"- {k}: {v}" for k, v in self.status.items()]
        lines += ["", f"Software and hardware: `environment.json`. Logs: `{self.logs}`."]
        _io.atomic_text("\n".join(lines) + "\n", self.path("REPORT.md"))
        self.status["report"] = "ok"
        print(f"[ ok ] report -> {self.path('REPORT.md')}")
        return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument(
        "--logs",
        default=os.path.join(ROOT, "logs"),
        help="directory of the per-step logs (default: logs/ next to main.py)",
    )
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="small graphs and sizes: checks the pipeline, not the numbers",
    )
    ap.add_argument("--paper", default=None,
                    help="where the paper step writes generated/ and figures_paper/ (default: <out>/paper), "
                         "e.g. a checkout of the manuscript")
    ap.add_argument("--manuscript", default=None,
                    help="the manuscript whose \\res keys the paper step checks (relative to --paper)")
    ap.add_argument(
        "--bench-families",
        nargs="+",
        default=["ER", "BA", "RGG", "WS"],
        choices=["ER", "BA", "RGG", "WS"],
        help="families of the runtime benchmark (average degree ~ 2.1 n^0.4 in each)",
    )
    ap.add_argument(
        "--bench-max-n",
        type=int,
        default=200000,
        help="largest graph of the runtime benchmark",
    )
    ap.add_argument(
        "--bench-graphs",
        type=int,
        default=100,
        help="sizes per family (log-spaced from n = 50)",
    )
    ap.add_argument(
        "--bench-budget-hours",
        type=float,
        default=5000.0,
        help="runtime benchmark: skip a phase whose predicted time exceeds this (per size)",
    )
    ap.add_argument(
        "--bench-mem-gb",
        type=float,
        default=None,
        help="runtime benchmark: memory available (default: the Slurm allocation)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=10000,
        help="edges per checkpointed chunk of exact evaluation",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Numba threads (-1 = all cores)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore step stamps (per-unit checkpoints are still reused)",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failed step",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="stream step output to the terminal",
    )
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in STEPS.items():
            print(f"{k:14s} {v}")
        return
    steps = a.steps or CORE + ["consistency", "paper", "report"]
    r = Runner(a)
    print(f"[main] steps: {' '.join(steps)}\n[main] output: {r.out}")
    t0 = time.time()
    for s in steps:
        ok = getattr(r, s)()
        if not ok and not a.keep_going and s != "report":
            print(
                "[main] stopping (use --keep-going to continue); re-run the same command to resume"
            )
            r.report()
            sys.exit(1)
    if "report" not in steps:
        r.report()
    print(
        f"[main] done in {time.time() - t0:.0f}s; see {os.path.join(r.out, 'REPORT.md')}"
    )


if __name__ == "__main__":
    main()
