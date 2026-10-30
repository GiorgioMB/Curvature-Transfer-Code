#!/usr/bin/env python3
"""Reproduce every experimental table, figure, and statistic with one command.

    python main.py                       # everything
    python main.py --quick               # small smoke run of the whole pipeline (minutes)
    python main.py --steps bounds        # selected steps only
    python main.py --list                # show the steps and what each one produces

All outputs go to ``results/`` (``--out``). Every step writes a log to
``results/logs/<step>.log``.

Resuming. Every step saves what it computes as soon as it is computed (per
graph, per chunk of edges, and per graph size), using atomic writes (tmp
file + rename). After a kill (e.g. a wall-time limit) just
run the same command again: finished steps are skipped through their stamp in
``results/.stamps``, and the interrupted step continues from its last saved
unit. ``--force`` ignores the stamps (the per-unit checkpoints are still
reused; delete ``results/`` to recompute from scratch).

Consistency. All graphs come from one registry (experiments/suite.py), all
per-edge quantities from one store (results/edges, written by edges.py), and
all intervals from one definition (curvature_transfer/intervals.py), so every
table, figure, and number refers to the same edges. The ``consistency`` step
verifies these invariants and writes ``results/CONSISTENCY.md``; it fails if a
check fails.

``results/REPORT.md`` maps every experimental artifact to the file that
reproduces it, and ``results/environment.json`` records the software and
hardware used. All randomness is seeded; outputs other than timings do not
depend on the number of threads.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
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

# Version of the timed computations (benchmark, screening). Their stamps include
# it, so that a step finished with older timing code is rerun (2: the 2-hop
# kernel, which also evaluates the TLW certificate).
TIMING_VERSION = 2

# Figures produced by the "distributions" and "benchmark" steps.
FIGURES = [
    "fig_scatter_cBF_cOR__er_n800_p0_0100125_2d_Hist.png",
    "fig_scatter_cBF_cOR__ws_n800_k10_b0_2_2d_Hist.png",
    "fig_scatter_cBF_cOR__ba_n1600_m2_2d_Hist.png",
    "fig_scatter_cBF_cOR__rg_n1600_r0_039894_2d_Hist.png",
    "fig_scatter_cBF_cOR__hrg_n800_R5_0_a1_0_T0_5_2d_Hist.png",
    "fig_scatter_cBF_cOR__sbm2_n1000_pin0_012018_pout0_004006_2d_Hist.png",
    *[
        f"fig_{c}_{v}__{t}.png"
        for c in ("cOR", "cBF")
        for v in ("hist", "quantile")
        for t in (
            "er_n800_p0_0100125",
            "hrg_n800_R5_0_a1_0_T0_0",
            "real_yeast",
        )
    ],
]
FIG_GRAPHS = OrderedDict([
    ("ER", "ER(800, 0.0100125)"),
    ("HRG", "HRG(800, 5.0, 1.0, 0.0)"),
    ("Yeast", "Yeast"),
])

STEPS = {
    "environment": "software/hardware record -> environment.json",
    "tests": "pytest suite: exact OT vs LP, theoretical bounds on random graphs",
    "examples": "closed-form values (Q_n, K_n, trees, grid) -> tables/examples.txt",
    "edges": "per-edge statistics, bounds, exact curvatures of every registry graph (chunked) -> edges/",
    "bounds": "bound comparison and ranking tables, from edges/ -> tables/bounds/",
    "bounds_legacy": "same tables with the preliminary BF normalization -> tables/bounds_legacy/",
    "screening": "certify-then-solve evaluation, from edges/ -> tables/screening/",
    "distributions": "per-edge CSVs + distribution figures, from edges/ -> edges/csv/",
    "benchmark": "runtime scaling, one figure per family -> benchmark/",
    "figures": "collect key figures into one directory -> figures/",
    "consistency": "cross-checks across bounds, screening, figures, and benchmarks -> CONSISTENCY.md",
    "report": "REPORT.md mapping experimental artifacts to files",
}
CORE = [
    "environment",
    "tests",
    "examples",
    "edges",
    "bounds",
    "screening",
    "distributions",
    "benchmark",
    "figures",
]


# --------------------------------------------------------- helpers
def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _pct(x):
    return f"{100 * x:.1f}" if _fin(x) else "--"


def _num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _atomic_csv(rows, fieldnames, path):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    _io.atomic_text(buf.getvalue(), path)


def _parse_bounds(rows):
    R = []
    for r in rows:
        o = {
            k: (v if k in ("graph", "tag", "group") else _num(r, k))
            for k, v in r.items()
        }
        o["E"] = int(o["E"])
        R.append(o)
    return R


def _benchmark_files(bdir):
    """{family: CSV with the largest max n} among benchmark_<family>_<min>_<max>.csv."""
    out = {}
    for f in os.listdir(bdir):
        if f.startswith("benchmark_") and f.endswith(".csv"):
            parts = f[:-4].split("_")
            if len(parts) == 4 and parts[3].isdigit():
                fam, mx = parts[1], int(parts[3])
                if fam not in out or mx > out[fam][0]:
                    out[fam] = (mx, os.path.join(bdir, f))
    return {k: v[1] for k, v in out.items()}


def run_consistency_checks(results_dir):
    """Cross-checks between raw files, bound tables, screening, figures, and benchmarks."""
    import compare_bounds as cb
    import suite

    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    bpath = os.path.join(results_dir, "tables", "bounds", "bound_comparison.csv")
    sdir = os.path.join(results_dir, "tables", "screening")
    spath = os.path.join(sdir, "screening_benchmark.csv")
    lpath = os.path.join(sdir, "screening_large.csv")
    bounds_rows = _read_csv(bpath) if os.path.exists(bpath) else None
    screen_rows = _read_csv(spath) if os.path.exists(spath) else None
    large_rows = _read_csv(lpath) if os.path.exists(lpath) else None

    R = _parse_bounds(bounds_rows) if bounds_rows else []
    if R:
        add(
            "no certified bound is violated on any edge (tolerance 1e-7)",
            sum(o["violations"] for o in R) == 0,
            f"largest excess {max(o['max_violation'] for o in R):.2e}",
        )
        add(
            "Theorem 4.3 (quadrangle-augmented JL bound) is never below Jost-Liu",
            sum(o["n_ours_below_JL"] for o in R) == 0,
        )
        names = [
            s.name for s in suite.REGISTRY if s.name in {o["graph"] for o in R}
        ]
        add(
            "bound tables list the graphs in registry order",
            [o["graph"] for o in R] == names,
        )

    if R and screen_rows:
        gb = {o["graph"]: o for o in R}
        S = {r["graph"]: r for r in screen_rows}
        add(
            "bound and screening tables cover the same graphs",
            set(gb) == set(S),
            f"only in bounds: {sorted(set(gb) - set(S))}; only in screening: {sorted(set(S) - set(gb))}",
        )
        bad_e = [g for g in set(S) & set(gb) if int(float(S[g]["E"])) != gb[g]["E"]]
        add(
            "|E| is the same in the bound and screening tables",
            not bad_e,
            f"differs on {bad_e}",
        )
        for iv, ck in (("bf", "cert_bf"), ("sharpened", "cert_sharp")):
            bad = []
            for g in set(S) & set(gb):
                av, ce = _num(S[g], f"sign_avoided_{iv}"), gb[g][ck]
                if av < ce - 1e-12 or float(_pct(av)) < float(_pct(ce)):
                    bad.append(f"{g} ({_pct(av)} < {_pct(ce)})")
            add(
                f"{iv} interval: avoided fraction for c_OR<0 >= certified-sign fraction (exact and as printed)",
                not bad,
                "; ".join(bad),
            )
    if large_rows:
        bad = [r["graph"] for r in large_rows if str(r.get("sign_correct")) != "True"]
        add(
            "certify-then-solve on the large networks returned the exact answer",
            not bad,
            f"wrong on {bad}",
        )

    if R:
        R_agg = [o for o in R if o["group"] in suite.AGGREGATE_GROUPS]
        more = [
            o
            for o in R_agg
            if float(_pct(o["cert_ours"])) > float(_pct(o["cert_prior"]))
        ]
        equal = [
            o for o in R_agg if _pct(o["cert_ours"]) == _pct(o["cert_prior"])
        ]
        fewer = len(R_agg) - len(more) - len(equal)
        add(
            "sign-certification counts add up to the number of aggregate graphs",
            len(more) + len(equal) + fewer == len(R_agg),
        )
        lo = [
            cb.fnum(o[k])
            for o in R
            for k in (
                "gap_up_JL",
                "gap_up_TLW",
                "gap_up_ours",
                "gap_Lup_prior",
                "gap_Lup_ours",
            )
        ]
        add("no table entry prints as -0.000", "-0.000" not in lo)
        figs = os.path.join(results_dir, "figures")
        for lab, name in FIG_GRAPHS.items():
            spec = suite.BY_NAME[name]
            ok = os.path.exists(
                os.path.join(figs, f"fig_cOR_hist__{spec.file_tag}.png")
            ) or not os.path.isdir(figs)
            add(
                f"distributional figure '{lab}' is drawn from registry graph {name} (file tag {spec.file_tag})",
                ok,
            )
        bdir = os.path.join(results_dir, "benchmark")
        if os.path.isdir(bdir):
            bfiles = _benchmark_files(bdir)
            sizes = {
                fam: [int(r["n"]) for r in _read_csv(p)]
                for fam, p in bfiles.items()
            }
            if "ER" in sizes:
                bad = [
                    f
                    for f, ns in sizes.items()
                    if sorted(ns) != sorted(sizes["ER"])
                ]
                add(
                    "runtime benchmark: every family covers the same sizes as ER",
                    not bad,
                    f"different sizes in {bad}",
                )
            for fam in bfiles:
                f = os.path.join(figs, f"runtime_scaling_{fam.lower()}.pdf")
                add(
                    f"runtime figure of {fam} exists (figures/runtime_scaling_{fam.lower()}.pdf)",
                    os.path.exists(f) or not os.path.isdir(figs),
                )
        man = os.path.join(results_dir, "edges", "csv", "manifest.json")
        if os.path.exists(man):
            with open(man) as fh:
                m = {r["name"]: r for r in json.load(fh)["runs"]}
            bad = [
                o["graph"]
                for o in R
                if o["graph"] in m and int(m[o["graph"]]["m"]) != o["E"]
            ]
            add(
                "figure CSVs have the same number of edges as the table rows",
                not bad,
                f"differs on {bad}",
            )

    return out


# -------------------------------------------------------------- runner
class Runner:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        for d in ("logs", ".stamps", "tables", "figures"):
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
        log = self.path("logs", f"{step}.log")
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
            self.status[step] = f"FAILED (see logs/{step}.log)"
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

    def bounds_legacy(self):
        return self.run(
            "bounds_legacy",
            [
                PY,
                "compare_bounds.py",
                "--varpi",
                "legacy",
                "--results",
                self.out,
                "--out",
                self.path("tables", "bounds_legacy"),
            ]
            + self.common(),
        )

    def screening(self):
        return self.run("screening", 
                        [PY, "screening.py", "--results", 
                             self.out, "--out", self.path("tables", "screening"),
                            "--data-dir", self.path("data", "large")] + self.common(),
                        cfg={"timing_version": TIMING_VERSION})

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
            if self.a.legacy_dir:
                cmd += ["--legacy-dir", os.path.abspath(self.a.legacy_dir)]
            ok &= self.run(f"benchmark_{fam}", cmd, cfg={"timing_version": TIMING_VERSION})
        return ok

    def figures(self):
        dst = self.path("figures")
        src = self.path("edges", "csv", "figures_paper")
        missing = []
        for f in FIGURES:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            else:
                missing.append(f)
        for fam in self.a.bench_families:
            rt = self.path(
                "benchmark", f"runtime_scaling_loglog_{fam.lower()}.pdf"
            )
            names = [f"runtime_scaling_{fam.lower()}.pdf"] + (
                ["runtime_scaling.pdf"] if fam == "ER" else []
            )
            if os.path.exists(rt):
                for nm in names:
                    shutil.copy2(rt, os.path.join(dst, nm))
            else:
                missing += names
        rt = self.path("benchmark", "runtime_scaling_all.pdf")
        if os.path.exists(rt):
            shutil.copy2(rt, os.path.join(dst, "runtime_scaling_all.pdf"))
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
        checks = run_consistency_checks(self.out)
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

    def report(self):
        rel = lambda *p: os.path.join(*p)
        ex = lambda *p: (
            "yes" if os.path.exists(self.path(*p)) else "**missing**"
        )
        bench = (
            "benchmark_ER_50_800.csv"
            if self.a.quick
            else f"benchmark_ER_50_{self.a.bench_max_n}.csv"
        )
        rows = [
            (
                "Per-edge arrays behind every table and figure",
                rel("edges"),
                "<tag>.paper.npz, <tag>.exact.npz",
            ),
            (
                "All per-graph bound quantities (CSV)",
                rel("tables", "bounds", "bound_comparison.csv"),
                "",
            ),
            (
                "Certify-then-solve on the registry graphs (stored intervals)",
                rel("tables", "screening", "screening_benchmark.csv"),
                "",
            ),
            (
                "Certify-then-solve on large real networks (Table 7, timed)",
                rel("tables", "screening", "screening_large.csv"),
                "",
            ),
            (
                "Cross-checks across tables, figures, and benchmarks",
                "CONSISTENCY.md",
                "",
            ),
            (
                "Closed-form values on structured graphs",
                rel("tables", "examples.txt"),
                "",
            ),
            (
                "Runtime scaling (ER)",
                rel("figures", "runtime_scaling.pdf"),
                "",
            ),
            (
                "Runtime scaling, one figure per family + 2x2 panel",
                rel("figures"),
                "runtime_scaling_*.pdf",
            ),
            (
                "Runtime data (ER; other families alongside)",
                rel("benchmark", bench),
                "benchmark_<family>_*.csv",
            ),
            (
                "Scatter / distributional views",
                rel("figures"),
                "fig_*.png",
            ),
            (
                "Per-edge CSVs behind distributional figures",
                rel("edges", "csv"),
                "*_edges.csv + manifest.json",
            ),
            (
                "Legacy BF normalization comparison",
                rel("tables", "bounds_legacy", "bound_comparison.csv"),
                "optional (--legacy-check)",
            ),
        ]
        lines = [
            "# Reproduction report",
            "",
            f"Generated {datetime.now(timezone.utc).isoformat()} by `{' '.join(sys.argv)}`"
            + (
                "  \n**Quick mode: reduced graphs and sizes.**"
                if self.a.quick
                else ""
            ),
            "",
            "| Artifact | File (under results/) | Present | Notes |",
            "| --- | --- | --- | --- |",
        ]
        for name, path, note in rows:
            lines.append(f"| {name} | `{path}` | {ex(path)} | {note} |")
        lines += (
            ["", "## Step status", ""]
            + [f"- {k}: {v}" for k, v in self.status.items()]
        )
        cons = self.path("CONSISTENCY.md")
        if os.path.exists(cons):
            with open(cons) as fh:
                lines += [
                    "",
                    fh.read()
                    .replace("# Consistency checks", "## Consistency checks")
                    .rstrip(),
                ]
        lines += [
            "",
            "Software and hardware: `environment.json`. Logs: `logs/`.",
        ]
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
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="small graphs and sizes: checks the pipeline, not the numbers",
    )
    ap.add_argument(
        "--legacy-check",
        action="store_true",
        help="also recompute the tables with the preliminary BF",
    )
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
        "--legacy-dir",
        default=None,
        help="preliminary repository, to add the old solver to the benchmark",
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
            opt = k == "bounds_legacy"
            print(f"{k:14s} {v}{'  [optional]' if opt else ''}")
        return
    steps = a.steps or (
        CORE[:5]
        + (["bounds_legacy"] if a.legacy_check else [])
        + CORE[5:]
        + ["consistency", "report"]
    )
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
