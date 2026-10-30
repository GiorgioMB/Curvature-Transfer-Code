"""Large real networks for the certify-then-solve experiment (Section 6.3).

These graphs (10^5 to 10^6 edges) are not part of :data:`suite.REGISTRY`: they
do not enter the bound tables, only the timed screening table. Each entry is
downloaded once from its public source into ``<data_dir>`` (default
``experiments/data/large``), parsed into an undirected simple graph (self-loops
dropped, directions and duplicates merged, node ids relabeled), and restricted
to its largest connected component by :func:`suite.largest_component`, as every
graph of the paper.

All graphs were chosen so that exact evaluation of every edge stays feasible
(maximum degree at most about 10^4): the speedup is measured against it, and
the certified answers are checked against it. The list spans two regimes: sparse
networks (average degree below ~20; an exact solve takes microseconds, so exact
evaluation of every edge takes seconds) and denser ones (average degree 15 to
76 with hubs; Amazon-Photo, Twitch-DE, Crocodile, Squirrel, Amazon-Computers,
GitHub), where the local transport problems are large and exact evaluation of
every edge is minutes of CPU time.

Download integrity: gzip/zip/npz files are checked by decompressing them, and
the parsed graph must have at least ``min_edges`` edges (well below the size
documented by the source); the raw and LCC sizes are written to the results.

Usage: python large_graphs.py [--data-dir DIR] [--only NAME;NAME]   (downloads and prints sizes)
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite  # noqa: E402

DATA_DIR = os.path.join(HERE, "data", "large")
SNAP = "https://snap.stanford.edu/data/"
GH = "https://raw.githubusercontent.com/"


@dataclass(frozen=True)
class LargeSpec:
    name: str           # name in the table
    tag: str            # file tag
    url: str
    fmt: str            # "snap" (whitespace edge list, '#' comments, .gz), "csv" (header id_1,id_2), "npz" (gnn-benchmark CSR)
    min_edges: int      # sanity check on the parsed (raw) graph
    source: str         # citation key / provenance, for the paper
    prose: str = ""

    @property
    def display(self) -> str:
        return self.prose or self.name


# Ordered by number of edges. Sizes in the comments are those documented by the sources.
LARGE: List[LargeSpec] = [
    LargeSpec("Deezer", "deezer_europe", GH + "benedekrozemberczki/karateclub/master/dataset/node_level/deezer/edges.csv",
              "csv", 90_000, "rozemberczki_characteristic_2020"),               # 28,281 nodes, 92,752 edges
    LargeSpec("Facebook", "facebook_pages", GH + "benedekrozemberczki/MUSAE/master/input/edges/facebook_edges.csv",
              "csv", 165_000, "rozemberczki_multiscale_2021"),                  # 22,470 nodes, 171,002 edges
    LargeSpec("Enron", "email_enron", SNAP + "email-Enron.txt.gz", "snap", 180_000,
              "leskovec_community_2009"),                                         # 36,692 nodes, 183,831 edges
    LargeSpec("AstroPh", "ca_astroph", SNAP + "ca-AstroPh.txt.gz", "snap", 190_000,
              "leskovec_graph_2007"),                                             # 18,772 nodes, 198,110 edges
    LargeSpec("Brightkite", "loc_brightkite", SNAP + "loc-brightkite_edges.txt.gz", "snap", 210_000,
              "cho_friendship_2011"),                                             # 58,228 nodes, 214,078 edges
    LargeSpec("Amazon-Photo", "amazon_photo", GH + "shchur/gnn-benchmark/master/data/npz/amazon_electronics_photo.npz",
              "npz", 115_000, "shchur_pitfalls_2018"),                          # 7,650 nodes, 119,081 edges
    LargeSpec("Twitch-DE", "twitch_de", GH + "benedekrozemberczki/MUSAE/master/input/edges/DE_edges.csv",
              "csv", 150_000, "rozemberczki_multiscale_2021"),                  # 9,498 nodes, 153,138 edges
    LargeSpec("Crocodile", "wiki_crocodile", GH + "benedekrozemberczki/MUSAE/master/input/edges/crocodile_edges.csv",
              "csv", 165_000, "rozemberczki_multiscale_2021", "Wikipedia-Crocodile"),  # 11,631 nodes, ~171,000 edges
    LargeSpec("Squirrel", "wiki_squirrel", GH + "benedekrozemberczki/MUSAE/master/input/edges/squirrel_edges.csv",
              "csv", 195_000, "rozemberczki_multiscale_2021", "Wikipedia-Squirrel"),   # 5,201 nodes, ~198,000 edges
    LargeSpec("Amazon-Comp.", "amazon_computers", GH + "shchur/gnn-benchmark/master/data/npz/amazon_electronics_computers.npz",
              "npz", 240_000, "shchur_pitfalls_2018", "Amazon-Computers"),      # 13,752 nodes, 245,861 edges
    LargeSpec("Coauthor-Phys.", "coauthor_physics", GH + "shchur/gnn-benchmark/master/data/npz/ms_academic_phy.npz",
              "npz", 240_000, "shchur_pitfalls_2018", "Coauthor-Physics"),       # 34,493 nodes, 247,962 edges
    LargeSpec("GitHub", "musae_github", GH + "benedekrozemberczki/MUSAE/master/input/edges/git_edges.csv",
              "csv", 285_000, "rozemberczki_multiscale_2021"),                  # 37,700 nodes, 289,003 edges
    LargeSpec("Epinions", "soc_epinions1", SNAP + "soc-Epinions1.txt.gz", "snap", 400_000,
              "richardson_trust_2003"),                                           # 75,879 nodes, 405,740 undirected edges
    LargeSpec("Amazon", "com_amazon", SNAP + "bigdata/communities/com-amazon.ungraph.txt.gz", "snap", 900_000,
              "yang_defining_2012"),                                              # 334,863 nodes, 925,872 edges
    LargeSpec("DBLP", "com_dblp", SNAP + "bigdata/communities/com-dblp.ungraph.txt.gz", "snap", 1_000_000,
              "yang_defining_2012"),                                              # 317,080 nodes, 1,049,866 edges
]
BY_NAME = {s.name: s for s in LARGE}
BY_TAG = {s.tag: s for s in LARGE}
QUICK = ["Deezer"]


def _download(url: str, path: str, retries: int = 4) -> None:
    tmp = path + ".part"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curvature-transfer/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            os.replace(tmp, path)
            return
        except Exception as e:  # network errors: retry with backoff
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def _relabel(e: np.ndarray) -> Tuple[int, np.ndarray]:
    e = np.asarray(e, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    uniq, inv = np.unique(e, return_inverse=True)
    e = inv.reshape(-1, 2).astype(np.int64)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return int(len(uniq)), e


def _parse(spec: LargeSpec, raw: bytes) -> Tuple[int, np.ndarray]:
    if spec.fmt == "snap":
        txt = gzip.decompress(raw) if spec.url.endswith(".gz") else raw
        e = np.loadtxt(io.BytesIO(txt), comments="#", dtype=np.int64, usecols=(0, 1))
    elif spec.fmt == "csv":
        e = np.loadtxt(io.BytesIO(raw), delimiter=",", skiprows=1, dtype=np.int64, usecols=(0, 1))
    elif spec.fmt == "npz":
        import scipy.sparse as sp
        z = np.load(io.BytesIO(raw), allow_pickle=True)
        A = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"])).tocoo()
        e = np.stack([A.row, A.col], axis=1)
    else:
        raise ValueError(spec.fmt)
    return _relabel(e)


def load(spec: LargeSpec, data_dir: Optional[str] = None) -> Tuple[int, np.ndarray, dict]:
    """(n, edges, info) of the largest connected component of ``spec``
    (downloaded and cached on first use)."""
    d = os.path.abspath(data_dir or DATA_DIR)
    os.makedirs(d, exist_ok=True)
    cache = os.path.join(d, f"{spec.tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return int(z["n"]), z["edges"], {"raw_nodes": int(z["raw_nodes"]), "raw_edges": int(z["raw_edges"])}
    base = os.path.basename(spec.url)
    ext = ".txt.gz" if base.endswith(".txt.gz") else os.path.splitext(base)[1]
    src = os.path.join(d, f"{spec.tag}.raw{ext}")  # by tag: several sources share a basename (edges.csv)
    if not os.path.exists(src):
        print(f"[large_graphs] downloading {spec.url}", flush=True)
        _download(spec.url, src)
    raw = open(src, "rb").read()
    n0, e0 = _parse(spec, raw)
    if len(e0) < spec.min_edges:
        raise RuntimeError(f"{spec.name}: parsed {len(e0)} edges, expected at least {spec.min_edges}; "
                           f"delete {src} and retry")
    n, e = suite.largest_component(n0, e0)
    np.savez_compressed(cache + ".tmp.npz", n=n, edges=e, raw_nodes=n0, raw_edges=len(e0))
    os.replace(cache + ".tmp.npz", cache)
    return n, e, {"raw_nodes": n0, "raw_edges": int(len(e0))}


def specs(only: Optional[Sequence[str]] = None, quick: bool = False) -> List[LargeSpec]:
    out = [s for s in LARGE if s.name in QUICK] if quick else list(LARGE)
    if only:
        keep = {x.strip() for x in only}
        out = [s for s in out if s.name in keep or s.tag in keep]
        missing = keep - {s.name for s in out} - {s.tag for s in out}
        if missing:
            raise SystemExit(f"unknown large graph(s): {sorted(missing)}; see experiments/large_graphs.py")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for s in specs(a.only.split(";") if a.only else None):
        n, e, info = load(s, a.data_dir)
        deg = np.bincount(e.ravel(), minlength=n)
        print(f"{s.name:16s} raw: {info['raw_nodes']:>9,} nodes {info['raw_edges']:>10,} edges | "
              f"LCC: {n:>9,} nodes {len(e):>10,} edges, max degree {deg.max():,}")


if __name__ == "__main__":
    main()
