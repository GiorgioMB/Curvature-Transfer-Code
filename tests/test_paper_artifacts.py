"""The paper step (experiments/paper_artifacts.py) on small synthetic results."""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, ".."))

import paper_artifacts as pa  # noqa: E402


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _results(tmp):
    res = os.path.join(tmp, "results")
    _write_csv(os.path.join(res, "tables", "screening", "screening_benchmark.csv"), [
        dict(graph="A", tag="a", group="random", E=100, sign_avoided_bf=0.2, sign_avoided_matching=0.5,
             sign_avoided_sharpened=0.6, sign_avoided_two_hop=0.55),
        dict(graph="B", tag="b", group="real", E=200, sign_avoided_bf=0.1, sign_avoided_matching=0.4,
             sign_avoided_sharpened=0.45, sign_avoided_two_hop=0.4),
    ])
    large = [dict(graph=name, tag=name.lower(), n=1000, E=E, max_deg=420, threads=4, time_exact=12.0,
                  sign_n_bf=E // 4, sign_n_two_hop=E // 4, sign_n_tlw=E // 10, sign_n_solved=E // 5,
                  sign_avoided=0.8, sign_time=12.0 / sp, sign_time_stage1=0.5, sign_speedup=sp, sign_correct=True)
             for name, E, sp in (("Deezer", 92752, 3.0), ("Facebook", 171002, 5.0))]
    _write_csv(os.path.join(res, "tables", "screening", "screening_large.csv"), large)
    bench = [dict(n=n, num_edges=4 * n, exact_s=0.01 * n, sharpened_interval_s=0.001 * n,
                  matching_interval_s=0.0012 * n, bf_interval_s=0.0001 * n, threads=4, chunked=False,
                  exact_status="ok", matching_interval_status="ok", sharpened_interval_status="ok")
             for n in (50, 400, 1600)]
    _write_csv(os.path.join(res, "benchmark", "benchmark_ER_50_1600.csv"), bench)
    os.makedirs(os.path.join(res, "screening_map"))
    open(os.path.join(res, "screening_map", "screening_map_numbers.tex"), "w").write(
        "% written by experiments/screening_map.py\n"
        "\\expandafter\\def\\csname res@MapEdges\\endcsname{2,985}\n"
        "\\expandafter\\def\\csname res@MapBF\\endcsname{52.8}\n")
    os.makedirs(os.path.join(res, "figures"))
    open(os.path.join(res, "figures", "runtime_scaling_er.pdf"), "w").write("%PDF")
    return res


def test_build_writes_numbers_rows_index_and_figures(tmp_path):
    res = _results(str(tmp_path))
    paper = os.path.join(str(tmp_path), "paper")
    os.makedirs(paper)
    open(os.path.join(paper, "manuscript.tex"), "w").write(
        "\\documentclass{article}\\begin{document}\\res{ScreenLargeCount} \\res{NumGraphs} \\res{MapEdges} "
        "\\resrows{screening} \\resrows{bound-comparison}\\end{document}\n")
    info = pa.build(res, paper)
    tex = open(os.path.join(paper, "generated", "numbers.tex")).read()
    for key, val in (("ScreenLargeCount", "2"), ("ScreenLargeMaxEdges", "171{,}002"), ("ScreenSignShareBFMin", "25.0"),
                     ("ScreenSignSpeedupMedian", "4.0"), ("ScreenThreads", "4"), ("ScreenBenchSignSharpMin", "45.0"),
                     ("BenchMinN", "50"), ("MapEdges", "2,985"), ("MapBF", "52.8")):
        assert f"\\csname res@{key}\\endcsname{{{val}}}" in tex, key
    assert info["tables"]["screening"] == "generated" and info["tables"]["runtime"] == "generated"
    assert info["tables"]["bound-comparison"].startswith("MISSING")
    assert info["missing"] == ["NumGraphs"] and info["missing_rows"] == ["bound-comparison"]
    assert "Deezer & 92{,}752 & 420" in open(os.path.join(paper, "generated", "rows-screening.tex")).read()
    assert os.path.exists(os.path.join(paper, "figures_paper", "runtime_scaling_er.pdf"))
    assert "NumGraphs" in open(os.path.join(paper, "generated", "INDEX.md")).read()


def test_consistency_checks_run_on_partial_results(tmp_path):
    checks = pa.consistency_checks(_results(str(tmp_path)))
    assert checks and all(ok for _, ok, _ in checks)
