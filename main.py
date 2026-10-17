#!/usr/bin/env python3
"""Reproduce every experimental table, figure, and statistic with one command.

    python main.py                       # everything except the GNN case study
    python main.py --gnn --delta 0.01    # also the rewiring case study (needs the [gnn] extra)
    python main.py --quick               # small smoke run of the whole pipeline (minutes)
    python main.py --steps bounds        # selected steps only
    python main.py --list                # show the steps and what each one produces

All outputs go to ``results/`` (``--out``). Every step writes a log to
``results/logs/<step>.log``.

Resuming. Every step saves what it computes as soon as it is computed (per
graph, per chunk of edges, per graph size, per rewiring iteration, per Optuna
trial, per CV fold, and, while a fold trains, every --ckpt-seconds), using
atomic writes (tmp file + rename). After a kill (e.g. a wall-time limit) just
run the same command again: finished steps are skipped through their stamp in
``results/.stamps``, and the interrupted step continues from its last saved
unit. ``--force`` ignores the stamps (the per-unit checkpoints are still
reused; delete ``results/`` to recompute from scratch).

Consistency. All graphs come from one registry (experiments/suite.py), all
per-edge quantities from one store (results/edges, written by edges.py), and
all intervals from one definition (curvature_transfer/intervals.py), so every
table, figure, and number refers to the same edges.

``results/REPORT.md`` maps every experimental artifact to the file that
reproduces it, and ``results/environment.json`` records the software and
hardware used. All randomness is seeded; outputs other than timings do not
depend on the number of threads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")
PY = sys.executable
sys.path.insert(0, ROOT)
from curvature_transfer import _io  # noqa: E402

# Figures produced by the "distributions" and "benchmark" steps.
FIGURES = [
    "fig_scatter_cBF_cOR__er_n800_p0_0100125_2d_Hist.png",
    "fig_scatter_cBF_cOR__ws_n800_k10_b0_2_2d_Hist.png",
    "fig_scatter_cBF_cOR__ba_n1600_m2_2d_Hist.png",
    "fig_scatter_cBF_cOR__rg_n1600_r0_039894_2d_Hist.png",
    "fig_scatter_cBF_cOR__hrg_n800_R5_0_a1_0_T0_5_2d_Hist.png",
    "fig_scatter_cBF_cOR__sbm2_n1000_pin0_012018_pout0_004006_2d_Hist.png",
    *[f"fig_{c}_{v}__{t}.png" for c in ("cOR", "cBF") for v in ("hist", "quantile")
      for t in ("er_n800_p0_0100125", "hrg_n800_R5_0_a1_0_T0_0", "real_yeast")],
]
GNN_DATASETS = ["ZINC", "QM9", "ogbg-molhiv", "minesweeper", "Tree-NeighborsMatch"]
GNN_ARCHS = ["GCN", "GIN", "GAT"]

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
    "gnn": "rewiring case study (needs torch) -> gnn/",
    "gnn_stats": "non-inferiority tests for the case study -> tables/gnn_stats.txt",
    "ablation": "SDRF runtime ablation on Minesweeper (needs torch) -> ablation/",
    "report": "REPORT.md mapping experimental artifacts to files",
}
CORE = ["environment", "tests", "examples", "edges", "bounds", "screening", "distributions", "benchmark", "figures"]


class Runner:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        for d in ("logs", ".stamps", "tables", "figures"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg")
        self.env["PYTHONPATH"] = ROOT + os.pathsep + self.env.get("PYTHONPATH", "")
        if a.jobs is not None:
            self.env["NUMBA_NUM_THREADS"] = str(os.cpu_count() if a.jobs == -1 else a.jobs)
        self.status = {}
        n = _io.cleanup_tmp(self.out)
        if n:
            print(f"[main] removed {n} leftover temporary file(s) from an interrupted run")

    def path(self, *p):
        return os.path.join(self.out, *p)

    def stamp(self, step, cfg):
        h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
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
        with open(log, "a") as fh:  # append: the log of an interrupted attempt is kept
            fh.write(f"\n===== {datetime.now(timezone.utc).isoformat()} {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                if self.a.verbose:
                    sys.stdout.write("       " + line)
            p.wait()
        dt = time.time() - t0
        if p.returncode != 0:
            print(f"[FAIL] {step} after {dt:.0f}s (exit {p.returncode}); see {log}")
            self.status[step] = f"FAILED (see logs/{step}.log)"
            return False
        _io.atomic_text(datetime.now(timezone.utc).isoformat(), st)
        print(f"[ ok ] {step} ({dt:.0f}s)")
        self.status[step] = f"ok ({dt:.0f}s)"
        return True

    # ------------------------------------------------------------- steps
    def environment(self):
        info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                "argv": sys.argv, "quick": self.a.quick, "seed": self.a.seed,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        for mod in ("numpy", "scipy", "numba", "networkx", "pandas", "matplotlib", "torch", "torch_geometric", "optuna", "sklearn"):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:
                info[mod] = None
        try:
            info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                                         stderr=subprocess.DEVNULL).strip()
        except Exception:
            info["git_commit"] = None
        _io.atomic_json(info, self.path("environment.json"))
        self.status["environment"] = "ok"
        return True

    def common(self):
        c = ["--seed", str(self.a.seed), "--chunk", str(self.a.chunk)] + (["--quick"] if self.a.quick else [])
        return c + ([] if self.a.jobs is None else ["--jobs", str(self.a.jobs)])

    def tests(self):
        return self.run("tests", [PY, "-m", "pytest", "-q", os.path.join(ROOT, "tests")], cwd=ROOT)

    def examples(self):
        return self.run("examples", [PY, "check_examples.py", "--out", self.path("tables", "examples.txt")])

    def edges(self):
        return self.run("edges", [PY, "edges.py", "--out", self.out] + self.common())

    def bounds(self):
        return self.run("bounds", [PY, "compare_bounds.py", "--results", self.out,
                                   "--out", self.path("tables", "bounds")] + self.common())

    def bounds_legacy(self):
        return self.run("bounds_legacy", [PY, "compare_bounds.py", "--varpi", "legacy", "--results", self.out,
                                          "--out", self.path("tables", "bounds_legacy")] + self.common())

    def screening(self):
        return self.run("screening", [PY, "screening.py", "--results", self.out,
                                      "--out", self.path("tables", "screening")] + self.common())

    def distributions(self):
        return self.run("distributions", [PY, "distributions.py", "--results", self.out] + self.common())

    def benchmark(self):
        ok = True
        mx, ng = (800, 6) if self.a.quick else (self.a.bench_max_n, self.a.bench_graphs)
        for fam in self.a.bench_families:
            cmd = [PY, "benchmark.py", "--family", fam, "--min-n", "50", "--max-n", str(mx), "--graphs", str(ng),
                   "--seed", "42", "--budget-hours", str(self.a.bench_budget_hours), "--out", self.path("benchmark")]
            cmd += [] if self.a.jobs is None else ["--jobs", str(self.a.jobs)]
            cmd += [] if self.a.bench_mem_gb is None else ["--mem-gb", str(self.a.bench_mem_gb)]
            if self.a.legacy_dir:
                cmd += ["--legacy-dir", os.path.abspath(self.a.legacy_dir)]
            ok &= self.run(f"benchmark_{fam}", cmd)
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
        for fam in self.a.bench_families:  # one runtime figure per family, plus the 2x2 panel
            rt = self.path("benchmark", f"runtime_scaling_loglog_{fam.lower()}.pdf")
            names = [f"runtime_scaling_{fam.lower()}.pdf"] + (["runtime_scaling.pdf"] if fam == "ER" else [])
            if os.path.exists(rt):
                for nm in names:
                    shutil.copy2(rt, os.path.join(dst, nm))
            else:
                missing += names
        rt = self.path("benchmark", "runtime_scaling_all.pdf")
        if os.path.exists(rt):
            shutil.copy2(rt, os.path.join(dst, "runtime_scaling_all.pdf"))
        if missing:
            print(f"[warn] figures not produced: {missing}" + (" (expected with --quick)" if self.a.quick else ""))
        self.status["figures"] = "ok" if not missing else f"ok, {len(missing)} missing"
        return True

    def gnn(self):
        if self.a.delta is None:
            print("[FAIL] gnn requires --delta (the non-inferiority margin, fixed before the experiments)")
            self.status["gnn"] = "FAILED (no --delta)"
            return False
        ok = True
        cfg = ({"reps": 2, "cv_split": 3, "trials": 2, "epochs": 5} if self.a.quick else
               {"reps": 50, "cv_split": 5, "trials": 50, "epochs": 200})
        archs = ["GCN"] if self.a.quick else GNN_ARCHS
        datasets = ["Tree-NeighborsMatch"] if self.a.quick else GNN_DATASETS
        cfg.update({"archs": archs, "datasets": datasets, "delta": self.a.delta, "seed": self.a.seed,
                    "max_iters_rewiring": {"minesweeper": 40, "other": 2 if self.a.quick else 5}})
        os.makedirs(self.path("gnn"), exist_ok=True)
        _io.atomic_json(cfg, self.path("gnn", "config.json"))
        for ds in datasets:
            for arch in archs:
                its = cfg["max_iters_rewiring"]["minesweeper" if ds == "minesweeper" else "other"]
                cmd = [PY, "gnn_experiments.py", "--dataset", ds, "--arch", arch, "--seed", str(self.a.seed),
                       "--out-dir", self.path("gnn"), "--workers", str(self.a.gnn_workers),
                       "--max-iters-rewiring", str(its), "--trials", str(cfg["trials"]), "--epochs", str(cfg["epochs"]),
                       "--reps", str(cfg["reps"]), "--cv-split", str(cfg["cv_split"]),
                       "--ckpt-seconds", str(self.a.ckpt_seconds), "--save-seconds", str(self.a.ckpt_seconds)]
                ok &= self.run(f"gnn_{ds}_{arch}", cmd)
                if not ok and not self.a.keep_going:
                    return False
        return ok

    def gnn_stats(self):
        gdir = self.path("gnn")
        if os.path.isdir(os.path.join(gdir, "parts")):
            sys.path.insert(0, EXP)
            import merge_results
            merge_results.merge(gdir)
        res = os.path.join(gdir, "gnn_results.json")
        if not os.path.exists(res):
            print("[skip] gnn_stats: no gnn results")
            self.status["gnn_stats"] = "skipped (no results)"
            return True
        if self.a.delta is None:
            print("[FAIL] gnn_stats requires --delta")
            return False
        # stamped with the content of the results, so new repetitions trigger a re-run
        digest = hashlib.sha256(open(res, "rb").read()).hexdigest()[:16]
        ok = self.run("gnn_stats", [PY, "stats.py", res, "--delta", str(self.a.delta)], cfg={"results": digest})
        if ok:
            shutil.copy2(self.path("logs", "gnn_stats.log"), self.path("tables", "gnn_stats.txt"))
        return ok

    def ablation(self):
        return self.run("ablation", [PY, "sdrf_ablation.py", "--seed", str(self.a.seed), "--out", self.path("ablation")])

    def report(self):
        rel = lambda *p: os.path.join(*p)
        ex = lambda *p: "yes" if os.path.exists(self.path(*p)) else "**missing**"
        bench = "benchmark_ER_50_800.csv" if self.a.quick else f"benchmark_ER_50_{self.a.bench_max_n}.csv"
        rows = [
            ("Per-edge arrays behind every table and figure", rel("edges"), "<tag>.paper.npz, <tag>.exact.npz"),
            ("Bound comparison, ranking, BF/sharpened (LaTeX rows)", rel("tables", "bounds", "bound_comparison_rows.tex"), ""),
            ("All per-graph quantities (CSV)", rel("tables", "bounds", "bound_comparison.csv"), ""),
            ("Certify-then-solve table", rel("tables", "screening", "screening.csv"), ""),
            ("Closed-form values on structured graphs", rel("tables", "examples.txt"), ""),
            ("Runtime scaling (ER)", rel("figures", "runtime_scaling.pdf"), ""),
            ("Runtime scaling, one figure per family + 2x2 panel", rel("figures"), "runtime_scaling_*.pdf"),
            ("Runtime data (ER; other families alongside)", rel("benchmark", bench), "benchmark_<family>_*.csv"),
            ("Scatter / distributional views", rel("figures"), "fig_*.png"),
            ("Per-edge CSVs behind distributional figures", rel("edges", "csv"), "*_edges.csv + manifest.json"),
            ("Legacy BF normalization comparison", rel("tables", "bounds_legacy", "bound_comparison_rows.tex"), "optional"),
            ("Downstream GNN rewiring results", rel("gnn", "gnn_results.json"), "optional (--gnn)"),
            ("GNN non-inferiority tests", rel("tables", "gnn_stats.txt"), "optional (--gnn)"),
        ]
        lines = ["# Reproduction report", "",
                 f"Generated {datetime.now(timezone.utc).isoformat()} by `{' '.join(sys.argv)}`"
                 + ("  \n**Quick mode: reduced graphs and sizes.**" if self.a.quick else ""), "",
                 "| Artifact | File (under results/) | Present | Notes |", "| --- | --- | --- | --- |"]
        for name, path, note in rows:
            lines.append(f"| {name} | `{path}` | {ex(path)} | {note} |")
        lines += ["", "## Step status", ""] + [f"- {k}: {v}" for k, v in self.status.items()]
        lines += ["", "Software and hardware: `environment.json`. Logs: `logs/`."]
        _io.atomic_text("\n".join(lines) + "\n", self.path("REPORT.md"))
        self.status["report"] = "ok"
        print(f"[ ok ] report -> {self.path('REPORT.md')}")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=None)
    ap.add_argument("--quick", action="store_true", help="small graphs and sizes: checks the pipeline, not the numbers")
    ap.add_argument("--gnn", action="store_true", help="also run the rewiring case study and its statistics")
    ap.add_argument("--ablation", action="store_true", help="also run the SDRF ablation (needs torch)")
    ap.add_argument("--legacy-check", action="store_true", help="also recompute the tables with the preliminary BF")
    ap.add_argument("--bench-families", nargs="+", default=["ER", "BA", "RGG", "WS"], choices=["ER", "BA", "RGG", "WS"],
                    help="families of the runtime benchmark (average degree ~ 2.1 n^0.4 in each)")
    ap.add_argument("--bench-max-n", type=int, default=200000, help="largest graph of the runtime benchmark")
    ap.add_argument("--bench-graphs", type=int, default=100, help="sizes per family (log-spaced from n = 50)")
    ap.add_argument("--bench-budget-hours", type=float, default=100.0,
                    help="runtime benchmark: skip a phase whose predicted time exceeds this (per size)")
    ap.add_argument("--bench-mem-gb", type=float, default=None,
                    help="runtime benchmark: memory available (default: the Slurm allocation)")
    ap.add_argument("--legacy-dir", default=None, help="preliminary repository, to add the old solver to the benchmark")
    ap.add_argument("--delta", type=float, default=None, help="non-inferiority margin for the GNN statistics")
    ap.add_argument("--gnn-workers", type=int, default=4)
    ap.add_argument("--ckpt-seconds", type=float, default=600,
                    help="GNN: seconds between training / rewiring checkpoints (lower = less lost work on a kill)")
    ap.add_argument("--chunk", type=int, default=10000, help="edges per checkpointed chunk of exact evaluation")
    ap.add_argument("--jobs", type=int, default=None, help="Numba threads (-1 = all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore step stamps (per-unit checkpoints are still reused)")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    ap.add_argument("--verbose", action="store_true", help="stream step output to the terminal")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in STEPS.items():
            opt = k in ("bounds_legacy", "gnn", "gnn_stats", "ablation")
            print(f"{k:14s} {v}{'  [optional]' if opt else ''}")
        return
    steps = a.steps or (CORE[:5] + (["bounds_legacy"] if a.legacy_check else []) + CORE[5:]
                        + (["gnn", "gnn_stats"] if a.gnn else []) + (["ablation"] if a.ablation else [])
                        + ["report"])
    r = Runner(a)
    print(f"[main] steps: {' '.join(steps)}\n[main] output: {r.out}")
    t0 = time.time()
    for s in steps:
        ok = getattr(r, s)()
        if not ok and not a.keep_going and s != "report":
            print("[main] stopping (use --keep-going to continue); re-run the same command to resume")
            r.report()
            sys.exit(1)
    if "report" not in steps:
        r.report()
    print(f"[main] done in {time.time() - t0:.0f}s; see {os.path.join(r.out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
