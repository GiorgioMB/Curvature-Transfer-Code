#!/usr/bin/env python3
"""Reproduce every table, figure, and number of the paper with one command.

    python main.py                 # everything except the GNN case study
    python main.py --quick         # small smoke run of the whole pipeline (minutes)
    python main.py --gnn --delta 0.01   # also the rewiring case study (needs the [gnn] extra)
    python main.py --steps bounds figures   # selected steps only
    python main.py --list          # show the steps and what each one produces

All outputs go to ``results/`` (``--out``). Every step writes a log to
``results/logs/<step>.log`` and a stamp to ``results/.stamps``; re-running
skips completed steps with the same configuration (``--force`` to redo them).
``results/REPORT.md`` maps every artifact of the paper to the file that
reproduces it, and ``results/environment.json`` records the software and
hardware used. All randomness is seeded; outputs do not depend on the number
of threads.
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

# Figures included by the manuscript (figures_paper/<name>), produced by the
# "distributions" and "benchmark" steps.
PAPER_FIGURES = [
    "fig_scatter_cBF_cOR__er_n800_p0_0100125_2d_Hist.png",
    "fig_scatter_cBF_cOR__ws_n800_k10_b0_2_2d_Hist.png",
    "fig_scatter_cBF_cOR__ba_n1600_m2_2d_Hist.png",
    "fig_scatter_cBF_cOR__rg_n1600_r0_039894_2d_Hist.png",
    "fig_scatter_cBF_cOR__hrg_n800_R5_0_a1_0_T0_5_2d_Hist.png",
    "fig_scatter_cBF_cOR__sbm2_n1000_pin0_012018_pout0_004006_2d_Hist.png",
    *[f"fig_{c}_{v}__{t}.png" for c in ("cOR", "cBF") for v in ("hist", "quantile")
      for t in ("er_n800_p0_0100125", "hrg_n800_R5_0_a1_0_T0_0", "real_yeast")],
]
PAPER = os.path.join(ROOT, "paper")
GNN_DATASETS = ["ZINC", "QM9", "ogbg-molhiv", "minesweeper", "Tree-NeighborsMatch"]
GNN_ARCHS = ["GCN", "GIN", "GAT"]

STEPS = {
    "environment": "software/hardware record -> environment.json",
    "tests": "pytest suite: exact OT vs LP, every theorem of the paper on random graphs",
    "examples": "closed-form values stated in the paper (Q_n, K_n, trees, grid) -> tables/examples.txt",
    "bounds": "Tables 3, 4, 5 and the ranking table (+ every aggregate number in the text) -> tables/bounds/",
    "bounds_legacy": "same tables with the preliminary BF normalization (consistency check vs arXiv v1) -> tables/bounds_legacy/",
    "screening": "certify-then-solve (Section 7.3) -> tables/screening/",
    "distributions": "per-edge tables of every graph family + Figures 2-3 -> runs/paper/",
    "benchmark": "runtime scaling, Figure 1 -> benchmark/",
    "figures": "collect the figures under the file names used by the manuscript -> figures_paper/",
    "gnn": "rewiring case study (Appendix A; needs torch) -> gnn/",
    "gnn_stats": "non-inferiority tests for the case study -> tables/gnn_stats.txt",
    "ablation": "SDRF runtime ablation on Minesweeper (needs torch) -> ablation/",
    "paper": "write paper/generated/ (all table rows + numbers.tex + INDEX.md), copy figures, compile paper/manuscript.pdf",
    "report": "REPORT.md mapping paper artifacts to files",
}
DEFAULT_STEPS = ["environment", "tests", "examples", "bounds", "screening", "distributions", "benchmark", "figures", "paper", "report"]


class Runner:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        for d in ("logs", ".stamps", "tables", "figures_paper"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg")
        if a.jobs is not None:
            self.env["NUMBA_NUM_THREADS"] = str(os.cpu_count() if a.jobs == -1 else a.jobs)
        self.status = {}

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
        with open(log, "w") as fh:
            p = subprocess.Popen(cmd, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                if self.a.verbose:
                    sys.stdout.write("       " + line)
            p.wait()
        dt = time.time() - t0
        if p.returncode != 0:
            print(f"[FAIL] {step} after {dt:.0f}s (exit {p.returncode}); see {log}")
            self.status[step] = f"FAILED (see logs/{step}.log)"
            return False
        open(st, "w").write(datetime.now(timezone.utc).isoformat())
        print(f"[ ok ] {step} ({dt:.0f}s)")
        self.status[step] = f"ok ({dt:.0f}s)"
        return True

    # ------------------------------------------------------------- steps
    def environment(self):
        info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                "argv": sys.argv, "quick": self.a.quick, "seed": self.a.seed}
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
        json.dump(info, open(self.path("environment.json"), "w"), indent=2)
        self.status["environment"] = "ok"
        return True

    def jobs(self):
        return [] if self.a.jobs is None else ["--jobs", str(self.a.jobs)]

    def tests(self):
        return self.run("tests", [PY, "-m", "pytest", "-q", os.path.join(ROOT, "tests")], cwd=ROOT)

    def examples(self):
        return self.run("examples", [PY, "check_examples.py", "--out", self.path("tables", "examples.txt")])

    def _quick_only(self):
        return ["--only", "Karate;BA(800, 2);Grid(40,40)", "--no-real"] if self.a.quick else []

    def bounds(self):
        return self.run("bounds", [PY, "compare_bounds.py", "--out", self.path("tables", "bounds"),
                                   "--seed", str(self.a.seed)] + self._quick_only() + self.jobs())

    def bounds_legacy(self):
        return self.run("bounds_legacy", [PY, "compare_bounds.py", "--varpi", "legacy", "--out",
                                          self.path("tables", "bounds_legacy"), "--seed", str(self.a.seed)]
                        + self._quick_only() + self.jobs())

    def screening(self):
        return self.run("screening", [PY, "screening.py", "--out", self.path("tables", "screening"),
                                      "--seed", str(self.a.seed)] + self._quick_only() + self.jobs())

    def distributions(self):
        preset = "tiny" if self.a.quick else "paper"
        return self.run("distributions", [PY, "run_experiments.py", "--preset", preset, "--phases", "distributions",
                                          "--figures", "--out-root", self.path("runs"), "--run-name", "paper",
                                          "--seed", str(self.a.seed)] + self.jobs())

    def benchmark(self):
        fams = ["ER", "BA", "RGG"] if self.a.all_families else ["ER"]
        ok = True
        for fam in fams:
            mx, ng = (800, 6) if self.a.quick else (15000, 100)  # the configuration of Figure 1
            cmd = [PY, "benchmark.py", "--family", fam, "--min-n", "50", "--max-n", str(mx), "--graphs", str(ng),
                   "--seed", "42", "--out", self.path("benchmark")] + self.jobs()
            if self.a.legacy_dir:
                cmd += ["--legacy-dir", os.path.abspath(self.a.legacy_dir)]
            ok &= self.run(f"benchmark_{fam}", cmd)
        return ok

    def figures(self):
        dst = self.path("figures_paper")
        src = self.path("runs", "paper", "figures_paper")
        os.makedirs(os.path.join(PAPER, "figures_paper"), exist_ok=True)
        missing = []
        for f in PAPER_FIGURES:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                shutil.copy2(os.path.join(src, f), os.path.join(PAPER, "figures_paper", f))
            else:
                missing.append(f)
        rt = self.path("benchmark", "runtime_scaling_loglog_er.pdf")
        if os.path.exists(rt):
            shutil.copy2(rt, os.path.join(dst, "runtime_scaling_loglog_er.pdf"))
            for d in (dst, os.path.join(PAPER, "figures_paper")):
                shutil.copy2(rt, os.path.join(d, "runtime_scaling.pdf"))  # name used by the manuscript
        else:
            missing.append("runtime_scaling.pdf")
        if missing and not self.a.quick:
            print(f"[warn] figures not produced: {missing}")
        self.status["figures"] = "ok" if not missing else f"ok, {len(missing)} missing (expected with --quick)"
        return True

    def gnn(self):
        if self.a.delta is None:
            print("[FAIL] gnn requires --delta (the non-inferiority margin, fixed before the experiments)")
            return False
        ok = True
        cfg = ({"reps": 2, "cv_split": 3, "trials": 2, "epochs": 5} if self.a.quick else
               {"reps": 50, "cv_split": 5, "trials": 50, "epochs": 200})
        cfg.update({"archs": ["GCN"] if self.a.quick else GNN_ARCHS, "delta": self.a.delta, "seed": self.a.seed,
                    "max_iters_rewiring": {"minesweeper": 40, "other": 2 if self.a.quick else 5}})
        os.makedirs(self.path("gnn"), exist_ok=True)
        json.dump(cfg, open(self.path("gnn", "config.json"), "w"), indent=2)
        for ds in (["Tree-NeighborsMatch"] if self.a.quick else GNN_DATASETS):
            for arch in (["GCN"] if self.a.quick else GNN_ARCHS):
                cmd = [PY, "gnn_experiments.py", "--dataset", ds, "--arch", arch, "--seed", str(self.a.seed),
                       "--out-dir", self.path("gnn"), "--workers", str(self.a.gnn_workers)]
                its = cfg["max_iters_rewiring"]["minesweeper" if ds == "minesweeper" else "other"]
                cmd += ["--max-iters-rewiring", str(its)]
                cmd += (["--trials", str(cfg["trials"]), "--epochs", str(cfg["epochs"]), "--reps", str(cfg["reps"]),
                         "--cv-split", str(cfg["cv_split"])] if self.a.quick else
                        ["--trials", str(cfg["trials"]), "--epochs", str(cfg["epochs"]), "--reps", str(cfg["reps"]),
                         "--cv-split", str(cfg["cv_split"])])
                ok &= self.run(f"gnn_{ds}_{arch}", cmd)
        return ok

    def gnn_stats(self):
        res = self.path("gnn", "gnn_results.json")
        if not os.path.exists(res):
            print("[skip] gnn_stats: no gnn results")
            self.status["gnn_stats"] = "skipped (no results)"
            return True
        ok = self.run("gnn_stats", [PY, "stats.py", res, "--delta", str(self.a.delta)])
        if ok:
            shutil.copy2(self.path("logs", "gnn_stats.log"), self.path("tables", "gnn_stats.txt"))
        return ok

    def ablation(self):
        return self.run("ablation", [PY, "sdrf_ablation.py", "--seed", str(self.a.seed), "--out", self.path("ablation")])

    def paper(self):
        sys.path.insert(0, EXP)
        import paper_artifacts
        status, N = paper_artifacts.build(self.out, PAPER)
        shutil.copy2(os.path.join(PAPER, "generated", "INDEX.md"), self.path("WHAT_GOES_WHERE.md"))
        missing = [k for k, v in status.items() if v != "generated"]
        print(f"[ ok ] paper/generated: {len(N.vals)} numbers, {len(status) - len(missing)}/{len(status)} tables"
              + (f" (missing: {', '.join(missing)})" if missing else ""))
        pdflatex, bibtex = shutil.which("pdflatex"), shutil.which("bibtex")
        if not (pdflatex and bibtex):
            self.status["paper"] = "generated; PDF not compiled (pdflatex/bibtex not found)"
            return True
        log = self.path("logs", "paper_latex.log")
        with open(log, "w") as fh:
            for cmd in ([pdflatex, "-interaction=nonstopmode", "manuscript.tex"], [bibtex, "manuscript"],
                        [pdflatex, "-interaction=nonstopmode", "manuscript.tex"],
                        [pdflatex, "-interaction=nonstopmode", "manuscript.tex"]):
                subprocess.run(cmd, cwd=PAPER, stdout=fh, stderr=subprocess.STDOUT)
        pdf = os.path.join(PAPER, "manuscript.pdf")
        if os.path.exists(pdf):
            shutil.copy2(pdf, self.path("manuscript.pdf"))
            self.status["paper"] = "ok: paper/manuscript.pdf (also results/manuscript.pdf)"
            print(f"[ ok ] compiled {pdf}")
        else:
            self.status["paper"] = "generated; LaTeX failed (see logs/paper_latex.log)"
        return True

    def report(self):
        rel = lambda *p: os.path.join(*p)
        ex = lambda *p: "yes" if os.path.exists(self.path(*p)) else "**missing**"
        rows = [
            ("Table 3 (non-lazy bound comparison)", rel("tables", "bounds", "bound_comparison_rows.tex"), "first block"),
            ("Table 4 (lazy bounds, sign certification)", rel("tables", "bounds", "bound_comparison_rows.tex"), "second block"),
            ("Ranking table", rel("tables", "bounds", "bound_comparison_rows.tex"), "third block"),
            ("Table 5 (BF and sharpened intervals)", rel("tables", "bounds", "bound_comparison_rows.tex"), "fourth block"),
            ("All per-graph quantities (CSV)", rel("tables", "bounds", "bound_comparison.csv"), ""),
            ("Numbers quoted in the abstract / Section 7", rel("tables", "bounds", "text_numbers.txt"), ""),
            ("Certify-then-solve table (Section 7.3)", rel("tables", "screening", "screening_rows.tex"), ""),
            ("Compiled paper with every number filled in", "manuscript.pdf", "step 'paper' (needs pdflatex)"),
            ("Where every table row and number goes", "WHAT_GOES_WHERE.md", "copy of paper/generated/INDEX.md"),
            ("Closed-form values stated in the paper", rel("tables", "examples.txt"), ""),
            ("Figure 1 (runtime scaling)", rel("figures_paper", "runtime_scaling.pdf"), ""),
            ("Figure 1 data", rel("benchmark", "benchmark_ER_50_15000.csv" if not self.a.quick else "benchmark_ER_50_800.csv"), ""),
            ("Figures 2-3 (scatter / distributional views)", rel("figures_paper"), "fig_*.png"),
            ("Per-edge tables behind Figures 2-3", rel("runs", "paper"), "*_edges.csv + manifest.json"),
            ("Consistency with arXiv v1 (legacy BF)", rel("tables", "bounds_legacy", "bound_comparison_rows.tex"), "optional step"),
            ("Appendix A: downstream results", rel("gnn", "gnn_results.json"), "optional step (--gnn)"),
            ("Appendix A: non-inferiority tests", rel("tables", "gnn_stats.txt"), "optional step (--gnn)"),
        ]
        lines = ["# Reproduction report", "",
                 f"Generated {datetime.now(timezone.utc).isoformat()} by `{' '.join(sys.argv)}`"
                 + ("  \n**Quick mode: reduced graphs and sizes, not the paper's numbers.**" if self.a.quick else ""), "",
                 "| Paper artifact | File (under results/) | Present | Notes |", "| --- | --- | --- | --- |"]
        for name, path, note in rows:
            lines.append(f"| {name} | `{path}` | {ex(path)} | {note} |")
        lines += ["", "## Step status", ""] + [f"- {k}: {v}" for k, v in self.status.items()]
        tn = self.path("tables", "bounds", "text_numbers.txt")
        if os.path.exists(tn):
            lines += ["", "## Numbers quoted in the text", "", "```", open(tn).read().rstrip(), "```"]
        lines += ["", "Software and hardware: `environment.json`. Logs: `logs/`."]
        open(self.path("REPORT.md"), "w").write("\n".join(lines) + "\n")
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
    ap.add_argument("--all-families", action="store_true", help="benchmark BA and RGG too (Figure 1 uses ER)")
    ap.add_argument("--legacy-dir", default=None, help="preliminary repository, to add the old solver to Figure 1")
    ap.add_argument("--delta", type=float, default=None, help="non-inferiority margin for the GNN statistics")
    ap.add_argument("--gnn-workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=None, help="Numba threads (-1 = all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="redo steps that are already done")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    ap.add_argument("--verbose", action="store_true", help="stream step output to the terminal")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in STEPS.items():
            print(f"{k:14s} {v}{'' if k in DEFAULT_STEPS else '  [optional]'}")
        return
    steps = a.steps or (DEFAULT_STEPS[:-1] + (["bounds_legacy"] if a.legacy_check else [])
                        + (["gnn", "gnn_stats"] if a.gnn else []) + (["ablation"] if a.ablation else []) + ["report"])
    r = Runner(a)
    print(f"[main] steps: {' '.join(steps)}\n[main] output: {r.out}")
    t0 = time.time()
    for s in steps:
        ok = getattr(r, s)()
        if not ok and not a.keep_going and s != "report":
            print("[main] stopping (use --keep-going to continue)")
            r.report()
            sys.exit(1)
    if "report" not in steps:
        r.report()
    print(f"[main] done in {time.time() - t0:.0f}s; see {os.path.join(r.out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
