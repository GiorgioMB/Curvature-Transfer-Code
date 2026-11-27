"""Paper artifacts from results/: generated/numbers.tex (every \\res and \\resrows key of the manuscript),
figures_paper/runtime_scaling_<family>.pdf, and generated/INDEX.md (what each number is and where it comes
from). Refuses to write if any validity or exactness check in the results failed."""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import suite   # noqa: E402

RESULTS = os.path.join(ROOT, "results")
GEN = os.path.join(ROOT, "generated")
FIG = os.path.join(ROOT, "figures_paper")


def _all(step):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, step, "*.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def f1(x):
    return f"{x:.1f}"


def f3(x):
    return f"{x:.3f}"


def fint(x):
    return f"{int(x):,}"


def fsec(x):
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def _clean_cpu(name):
    """'Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz' -> 'Intel Xeon Gold 6248'; 'AMD EPYC 7H12 64-Core Processor'
    -> 'AMD EPYC 7H12'."""
    for junk in ("(R)", "(TM)", "(tm)", " CPU", " Processor"):
        name = name.replace(junk, "")
    name = re.sub(r"@.*$", "", name)
    name = re.sub(r"\b\d+-Core\b", "", name)
    return " ".join(name.split())


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _months(starts):
    import datetime as dt
    a, b = (dt.datetime.fromtimestamp(x) for x in (min(starts), max(starts)))
    fa, fb = a.strftime("%B %Y"), b.strftime("%B %Y")
    if fa == fb:
        return fa
    return f"{a.strftime('%B')}--{fb}" if a.year == b.year else f"{fa}--{fb}"


def compute_keys():
    """Keys of the code availability statement from the provenance of every task (results/_tasks)."""
    recs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "_tasks", "*.json"))):
        with open(p) as fh:
            recs.append(json.load(fh))
    if not recs:
        return {}, [], []
    keys, problems = {}, []
    core_h = sum(r["wall"] * r["cores"] for r in recs) / 3600
    cpu_h = sum(r["cpu"] for r in recs) / 3600
    keys["NumTasks"] = fint(len(recs))
    keys["CoreHours"] = fint(round(core_h)) if core_h >= 100 else f"{core_h:.1f}"
    keys["CPUHours"] = fint(round(cpu_h)) if cpu_h >= 100 else f"{cpu_h:.1f}"
    keys["MaxTaskHours"] = f"{max(r['wall'] for r in recs) / 3600:.1f}"
    mem = max(r["maxrss_gb"] for r in recs)
    keys["PeakMemGB"] = f"{mem:.1f}" if mem < 10 else f"{mem:.0f}"
    clusters = sorted({r["cluster"] for r in recs if r["cluster"]})
    keys["ClusterName"] = _join([c.capitalize() for c in clusters]) if clusters else "a single machine"
    keys["RunMonth"] = _months([r["start"] for r in recs])
    keys["CPUModels"] = _join(sorted({_clean_cpu(r["cpu_model"]) for r in recs}))
    timing = [r for r in recs if r["task"][0] in ("kstar", "screening", "benchmark")]
    if timing:
        keys["TimingCPU"] = _join(sorted({_clean_cpu(r["cpu_model"]) for r in timing}))
        th = sorted({r["threads"] for r in timing}); rp = sorted({r["reps"] for r in timing})
        keys["Threads"] = _join([str(x) for x in th]); keys["Reps"] = _join([str(x) for x in rp])
        if len(th) > 1 or len(rp) > 1:
            problems.append(f"timing tasks used different threads {th} or repetitions {rp}")
    for name, key in (("python", "PythonVersion"), ("numpy", "NumpyVersion"), ("scipy", "ScipyVersion"),
                      ("numba", "NumbaVersion"), ("orcut", "OrcutVersion")):
        vs = sorted({r["versions"][name] for r in recs})
        keys[key] = vs[0]
        if len(vs) > 1:
            problems.append(f"tasks ran with different {name} versions: {vs}")
    index = [f"NumTasks, CoreHours (allocated cores x wall time), CPUHours (used), MaxTaskHours, PeakMemGB, "
             f"ClusterName, RunMonth, CPUModels, TimingCPU (kstar, screening, benchmark), Threads, Reps, and the "
             f"versions: the provenance of the {len(recs)} tasks in results/_tasks"]
    return keys, index, problems


def build(table_groups=("random", "real")):
    keys, rows, index, problems = {}, {}, [], []
    order = {s.tag: t for t, s in enumerate(suite.REGISTRY)}
    prose = {s.tag: (s.prose or s.name) for s in suite.REGISTRY}

    # ---- tightness and full range (Table 2)
    tight = sorted((r for r in _all("tightness") if r["group"] in table_groups), key=lambda r: order.get(r["tag"], 999))
    rng = {r["tag"]: r for r in _all("range")}
    for r in _all("tightness"):
        bad = {k: v for k, v in r["violations"].items() if v}
        if bad:
            problems.append(f"tightness {r['name']}: violations {bad}")
    for r in rng.values():
        if r["violations"]:
            problems.append(f"range {r['name']}: {r['violations']} violations of the chain")
    lines = []
    for r in tight:
        fr = rng.get(r["tag"])
        tail = f"{f1(fr['range_sign'])} & {f3(fr['range_width'])}" if fr and fr["complete"] else "-- & --"
        lines.append(f"{prose[r['tag']]} & {fint(r['m'])} & {f1(r['bm_sign'])} & {f3(r['bm_width'])} & "
                     f"{f1(r['tlw_sign'])} & {f3(r['tlw_width'])} & {f1(r['tq_sign'])} & {f3(r['tq_width'])} & "
                     f"{tail} \\\\")
    rows["TightnessRows"] = "\n".join(lines)
    if tight:
        keys["NumSynthetic"] = str(sum(r["group"] == "random" for r in tight))
        keys["NumReal"] = str(sum(r["group"] == "real" for r in tight))
        keys["NumGraphs"] = str(len(tight))
        keys["TQSignMin"] = f1(min(r["tq_sign"] for r in tight)); keys["TQSignMax"] = f1(max(r["tq_sign"] for r in tight))
        for k_, f_ in (("BM", "bm_sign"), ("TLW", "tlw_sign")):
            keys[f"{k_}SignMin"] = f1(min(r[f_] for r in tight)); keys[f"{k_}SignMax"] = f1(max(r[f_] for r in tight))
        index.append("NumSynthetic, NumReal, NumGraphs, TQSign*, BMSign*, TLWSign*: results/tightness, the non-lazy "
                     f"measure on the Table 2 graphs (groups {', '.join(table_groups)}); sign = share of edges whose "
                     "sign the interval settles, decided exactly on the grid 1/(d_i d_j)")
    comp = [rng[r["tag"]] for r in tight if r["tag"] in rng and rng[r["tag"]]["complete"]]
    if comp:
        keys["GapLoMean"] = f3(np.mean([c["gap_lo"] for c in comp]))
        keys["GapHiMean"] = f3(np.mean([c["gap_hi"] for c in comp]))
        keys["SignGapMax"] = f1(max(c["range_sign"] - c["tq_sign"] for c in comp))
        index.append(f"GapLoMean, GapHiMean: non-lazy measure; mean over the {len(comp)} graphs with a complete full range of the "
                     "per-graph mean gap between TQ and the ends of the full range; SignGapMax: largest difference "
                     "of settled signs, in percentage points (results/range)")

    # ---- other measures
    meas = _all("measures")
    if meas:
        gl, gh, sg = [], [], []
        for name in ("lazy", "half", "degree"):
            ok = [m[name] for m in meas if m[name]["complete"]]
            for m in meas:
                if m[name]["violations"]:
                    problems.append(f"measures {m['name']} {name}: {m[name]['violations']} violations")
            if ok:
                gl.append(np.mean([o["gap_lo"] for o in ok])); gh.append(np.mean([o["gap_hi"] for o in ok]))
                sg.append(max(o["range_sign"] - o["tq_sign"] for o in ok))
        keys["MeasGapLo"] = f3(max(gl)); keys["MeasGapHi"] = f3(max(gh)); keys["MeasSignGap"] = f1(max(sg))
        index.append(f"MeasGapLo, MeasGapHi, MeasSignGap: largest over the lazy measure, idleness 1/2 (certified "
                     f"at w_j and lifted by Corollary 5.4) and the degree-biased walk of the mean gaps and of the "
                     f"sign difference, on {len(meas)} graphs (results/measures)")

    # ---- the kappa* stage
    ks = [k for k in _all("kstar") if k.get("undecided", 0) >= 100]
    if ks:
        keys["KStarDecMin"] = f1(min(k["decided_by_kstar"] for k in ks))
        keys["KStarDecMax"] = f1(max(k["decided_by_kstar"] for k in ks))
        keys["KStarSlowMin"] = f1(min(k["slowdown"] for k in ks))
        keys["KStarSlowMax"] = f1(max(k["slowdown"] for k in ks))
        index.append(f"KStarDec*, KStarSlow*: {len(ks)} graphs with at least 100 edges left undecided by TQ; share of "
                     "them decided by the lower end and the slot-free upper end of the full range, and the time of "
                     "TQ -> full range -> orcut over that of TQ -> orcut (solve times; building the local graphs is "
                     "not counted, which favours the middle stage) (results/kstar)")

    # ---- screen-then-solve (Table 3)
    scr = sorted(_all("screening"), key=lambda r: r["m"])
    for r in scr:
        if not r["exact_answer"]:
            problems.append(f"screening {r['name']}: the answer differs from full evaluation")
    if scr:
        rows["ScreeningRows"] = "\n".join(
            f"{r['name']} & {fint(r['m'])} & {fint(r['dmax'])} & {fsec(r['t_full'])} & {fsec(r['t_tq'])} & "
            f"{f1(r['decided'])} & {f1(r['speedup'])}$\\times$ \\\\" for r in scr)
        keys["MaxEdges"] = fint(max(r["m"] for r in scr))
        keys["TQFracMin"] = f1(min(r["tq_frac"] for r in scr)); keys["TQFracMax"] = f1(max(r["tq_frac"] for r in scr))
        keys["DecMin"] = f1(min(r["decided"] for r in scr)); keys["DecMax"] = f1(max(r["decided"] for r in scr))
        sp = [r["speedup"] for r in scr]
        keys["CTSSpeedMin"] = f1(min(sp)); keys["CTSSpeedMax"] = f1(max(sp)); keys["CTSSpeedMed"] = f1(float(np.median(sp)))
        index.append(f"ScreeningRows, MaxEdges, TQFrac*, Dec*, CTSSpeed*: {len(scr)} networks (results/screening); "
                     "full evaluation and the solve stage use orcut with integer fractions; times are medians")

    # ---- where, when and with what the experiments ran (code availability statement)
    ck, cidx, cprob = compute_keys()
    keys.update(ck); index.extend(cidx); problems.extend(cprob)
    return keys, rows, index, problems


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    names = {"er": "Erd\u0151s\u2013R\u00e9nyi", "ba": "Barab\u00e1si\u2013Albert", "rgg": "random geometric",
             "ws": "Watts\u2013Strogatz"}
    made = []
    for b in _all("benchmark"):
        pts = b["points"]
        if not pts:
            continue
        m = [p["m"] for p in pts]
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        ax.loglog(m, [p["t_full"] for p in pts], "o-", label="full evaluation (orcut)")
        ax.loglog(m, [p["t_tq"] for p in pts], "s-", label="TQ")
        ax.loglog(m, [p["t_bf"] for p in pts], "^-", label="BF")
        ax.set_xlabel("$|E|$"); ax.set_ylabel("time (s)"); ax.set_title(names.get(b["family"], b["family"]), fontsize=9)
        ax.grid(True, which="both", lw=0.3, alpha=0.5); ax.legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        path = os.path.join(FIG, f"runtime_scaling_{b['family']}.pdf")
        fig.savefig(path); plt.close(fig)
        made.append(path)
    return made


def write(allow_problems=False):
    keys, rows, index, problems = build()
    if problems and not allow_problems:
        raise SystemExit("refusing to write the paper's numbers:\n  " + "\n  ".join(problems))
    os.makedirs(GEN, exist_ok=True)
    out = ["% generated by experiments/tq_artifacts.py; do not edit"]
    for k in sorted(keys):
        out.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{keys[k]}}}")
    for k in sorted(rows):
        out.append(f"\\expandafter\\def\\csname resrows@{k}\\endcsname{{%\n{rows[k].rstrip()}\n}}")
    with open(os.path.join(GEN, "numbers.tex"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    figs = figures()
    with open(os.path.join(GEN, "INDEX.md"), "w") as fh:
        fh.write("# Provenance of the numbers in generated/numbers.tex\n\n")
        for line in index:
            fh.write(f"- {line}\n")
        fh.write("\n## Values\n\n| Key | Value |\n|---|---|\n")
        for k in sorted(keys):
            fh.write(f"| {k} | {keys[k]} |\n")
        fh.write(f"\nFigures: {', '.join(os.path.relpath(f, ROOT) for f in figs) or 'none'}\n")
    return keys, rows, figs, problems
