"""CurvatureEngine: per-edge curvatures, bounds, and certified intervals.

>>> from curvature_transfer import CurvatureEngine
>>> eng = CurvatureEngine.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
>>> eng.bf()                      # Balanced Forman curvature, one value per edge
>>> eng.exact()["c_OR"]           # exact lazy Ollivier--Ricci curvature
>>> iv = eng.intervals()          # BF interval and sharpened interval
>>> iv["bf_lower"], iv["bf_upper"]

Edges are the undirected edges of ``eng.graph`` in ``u < v`` lexicographic
order; every returned array is aligned with ``eng.edges``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from . import _kernels as K
from . import bounds as B
from .graph import Graph

__all__ = ["CurvatureEngine"]


@contextmanager
def _threads(n_jobs: Optional[int]):
    """Temporarily set the number of Numba threads (scikit-learn semantics)."""
    import numba

    prev = numba.get_num_threads()
    if n_jobs is None:
        k = prev
    else:
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        cpus = numba.config.NUMBA_NUM_THREADS
        k = cpus if n_jobs == -1 else (n_jobs if n_jobs > 0 else max(1, cpus + 1 + n_jobs))
        k = max(1, min(k, cpus))
    numba.set_num_threads(k)
    try:
        yield
    finally:
        numba.set_num_threads(prev)


class CurvatureEngine:
    """Compute curvatures and bounds on every undirected edge of a graph.

    Parameters
    ----------
    graph : Graph, PyG ``Data``, networkx graph, or (M, 2) edge array.
    n_jobs : number of threads for the Numba kernels (None = Numba default,
        -1 = all, k > 0 = k threads, k < -1 = all but |k| - 1).
    varpi : ``"paper"`` (default) uses the 4-cycle normalization of Topping et
        al. / the paper (closers of k in U_i inside U_j). ``"legacy"``
        reproduces the preliminary code, which maximized over all of
        N(i)\\{j} into N(j)\\{i}; this underestimates the 4-cycle term on some
        edges (the bounds stay valid, but the BF values are not Topping's BF).
    """

    def __init__(self, graph, n_jobs: Optional[int] = None, varpi: str = "paper"):
        if varpi not in ("paper", "legacy"):
            raise ValueError("varpi must be 'paper' or 'legacy'")
        self.graph = _as_graph(graph)
        self.n_jobs = n_jobs
        self.varpi = varpi
        self._cache: Dict[str, object] = {}

    # ------------------------------------------------------------ builders
    @classmethod
    def from_edges(cls, edges, num_nodes=None, **kw) -> "CurvatureEngine":
        return cls(Graph.from_edges(edges, num_nodes), **kw)

    @property
    def edges(self) -> np.ndarray:
        return self.graph.edges

    @property
    def num_edges(self) -> int:
        return self.graph.num_edges

    # --------------------------------------------------------- statistics
    def stats(self) -> Dict[str, np.ndarray]:
        """2-hop statistics (cached): deg_i, deg_j, triangle, nUi, nUj, Xi,
        varpi, varpi_legacy (if requested), matching, C4, and c_BF."""
        if "stats" in self._cache:
            return self._cache["stats"]  # type: ignore[return-value]
        g = self.graph
        with _threads(self.n_jobs):
            raw = K.edge_stats_all(g.indptr, g.indices, g.edges, self.varpi == "legacy")
        di = g.deg[g.edges[:, 0]].astype(float)
        dj = g.deg[g.edges[:, 1]].astype(float)
        st = {
            "deg_i": di, "deg_j": dj,
            "triangle": raw[:, 0].astype(float),
            "nUi": raw[:, 1].astype(float), "nUj": raw[:, 2].astype(float),
            "Xi": raw[:, 3].astype(float),
            "varpi": raw[:, 4].astype(float),
            "matching": raw[:, 6].astype(float),
        }
        vp = raw[:, 5] if self.varpi == "legacy" else raw[:, 4]
        if self.varpi == "legacy":
            st["varpi_legacy"] = raw[:, 5].astype(float)
        st["sho_max"] = vp * np.maximum(di, dj)
        st["C4"] = B.c4_term(st["Xi"], vp, di, dj)
        st["c_BF"] = B.bf_curvature(di, dj, st["triangle"], st["C4"])
        self._cache["stats"] = st
        return st

    def bf(self) -> np.ndarray:
        return self.stats()["c_BF"]

    # ------------------------------------------------------ exact curvatures
    def exact(self, which: Sequence[str] = ("c_OR", "c_OR0", "c_LLY"),
              edges: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Exact curvatures by exact optimal transport (see ``_kernels``).

        ``which`` is any subset of {"c_OR", "c_OR0", "c_LLY"}; the TLW dual
        upper bounds ("tlw_dual_lazy", "tlw_dual_nonlazy") come for free with
        the corresponding measure. With ``edges`` (an array of edge ids) only
        those edges are evaluated and the result is aligned with ``edges``.
        """
        which = tuple(which)
        bad = set(which) - {"c_OR", "c_OR0", "c_LLY"}
        if bad:
            raise ValueError(f"unknown curvature(s): {sorted(bad)}")
        g = self.graph
        flags = ("c_OR" in which, "c_OR0" in which, "c_LLY" in which)
        import numba

        if edges is None:
            key = ("exact",) + flags
            if key not in self._cache:
                ids = np.arange(g.num_edges, dtype=np.int64)
                with _threads(self.n_jobs):
                    nch = 8 * numba.get_num_threads()
                    self._cache[key] = K.exact_ids(g.indptr, g.indices, g.edges, ids, *flags, nch)
            raw = self._cache[key]
        else:
            ids = np.ascontiguousarray(edges, dtype=np.int64)
            with _threads(self.n_jobs):
                nch = 8 * numba.get_num_threads()
                raw = K.exact_ids(g.indptr, g.indices, g.edges, ids, *flags, nch)
        out = {}
        if flags[0]:
            out["c_OR"] = raw[:, 0]
            out["tlw_dual_lazy"] = raw[:, 3]
        if flags[1]:
            out["c_OR0"] = raw[:, 1]
            out["tlw_dual_nonlazy"] = raw[:, 4]
        if flags[2]:
            out["c_LLY"] = raw[:, 2]
        return out

    def tlw_dual(self, edges: Optional[np.ndarray] = None) -> np.ndarray:
        """TLW dual upper bound on the lazy curvature (needs local distances,
        but no transport solve)."""
        import numba

        g = self.graph
        ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
        with _threads(self.n_jobs):
            return K.tlw_lazy_ids(g.indptr, g.indices, g.edges, ids, 8 * numba.get_num_threads())

    def prior_upper(self) -> np.ndarray:
        """Best prior upper certificate: min(TLW dual, zero-cost bound) (cached)."""
        if "prior_upper" not in self._cache:
            self._cache["prior_upper"] = np.minimum(self.tlw_dual(), self.bounds()["zero_cost_upper"])
        return self._cache["prior_upper"]

    # ------------------------------------------------------------- bounds
    def bounds(self) -> Dict[str, np.ndarray]:
        """All combinatorial bounds that need no transport solve (cached)."""
        if "bounds" in self._cache:
            return self._cache["bounds"]  # type: ignore[return-value]
        st = self.stats()
        di, dj, t = st["deg_i"], st["deg_j"], st["triangle"]
        S, T, Kk = B.degree_terms(di, dj)
        bf = st["c_BF"]
        Lsq = B.matching_lower(di, dj, t, st["matching"])
        A, Bd = B.bf_deficits(di, dj, t, st["C4"])
        const, slope = B.theta_const_slope(di, dj)
        out = {
            # non-lazy (c_OR-0)
            "jl_lower": B.jl_lower(di, dj, t),
            "jl_upper": B.jl_upper(di, dj, t),
            "tlw_lower": B.tlw_lower(di, dj, t, st["nUi"], st["nUj"]),
            "matching_lower": Lsq,
            "envelope_upper_nonlazy": B.envelope_upper(di, dj, t, st["nUi"], st["nUj"], st["Xi"], 0.0, 0.0),
            # lazy (c_OR)
            "jl_lower_lazy": B.lift(B.jl_lower(di, dj, t), di, dj),
            "matching_lower_lazy": B.lift(Lsq, di, dj),
            "envelope_upper": B.envelope_upper(di, dj, t, st["nUi"], st["nUj"], st["Xi"]),
            "zero_cost_upper": B.zero_cost_upper(di, dj, t),
            "theta_const": const, "theta_slope": slope, "theta_at_t": const + slope * t,
            # BF -> OR (Section 5)
            "bf_lower": B.phi_bf_to_or(bf, di, dj),
            "bf_upper": B.psi_bf_to_or(bf, di, dj),
            "bf_lower_preliminary": B.phi_bf_to_or_preliminary(bf, di, dj, st["C4"]),
            "bf_deficit_A": A, "bf_deficit_B": Bd,
            "bf_clipped": B.bf_clipped(bf, A, Bd),
            "sharpened_lower": B.sharpened_lower(Lsq, bf, di, dj),
        }
        out["sharpened_upper"] = np.minimum(out["envelope_upper"], out["bf_upper"])
        self._cache["bounds"] = out
        return out

    def intervals(self, prior_upper: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Certified intervals for c_OR.

        * BF interval (Corollary 5.6): depends on c_BF and the degrees only.
        * matching interval: [Lambda(L^square), U^env] (Section 6.1).
        * sharpened interval (Proposition 5.7): [Lambda(max(L^square, c_BF)), U^env].
        ``prior_upper`` (e.g. the TLW dual bound) is intersected with the upper
        ends of the matching and sharpened intervals when given.
        """
        b = self.bounds()
        up = b["envelope_upper"] if prior_upper is None else np.minimum(b["envelope_upper"], prior_upper)
        return {
            "bf_lower": b["bf_lower"], "bf_upper": b["bf_upper"],
            "matching_lower": b["matching_lower_lazy"], "matching_upper": up,
            "sharpened_lower": b["sharpened_lower"], "sharpened_upper": np.minimum(up, b["bf_upper"]),
        }

    def certified_intervals(self, prior_upper: Optional[np.ndarray] = None):
        """The BF, matching, and sharpened intervals of Section 6.1, as
        ``{kind: (lower, upper)}`` (see :mod:`curvature_transfer.intervals`).
        ``prior_upper`` defaults to :meth:`prior_upper` (TLW dual + zero-cost)."""
        from . import intervals as I
        return I.from_arrays(self.bounds(), self.prior_upper() if prior_upper is None else prior_upper)

    # ------------------------------------------------ transfer conveniences
    def bounds_from_BF(self, c_BF=None) -> Dict[str, np.ndarray]:
        """Lower/upper bounds on c_OR from BF values (default: this graph's BF)."""
        st = self.stats()
        z = st["c_BF"] if c_BF is None else self._edgewise(c_BF, "c_BF")
        return {"c_BF": z,
                "c_OR_lower_from_c_BF": B.phi_bf_to_or(z, st["deg_i"], st["deg_j"]),
                "c_OR_upper_from_c_BF": B.psi_bf_to_or(z, st["deg_i"], st["deg_j"])}

    def bounds_from_OR(self, c_OR) -> Dict[str, np.ndarray]:
        """Lower/upper bounds on c_BF from c_OR values (Corollaries C.2, C.3)."""
        st = self.stats()
        th = self._edgewise(c_OR, "c_OR")
        return {"c_OR": th,
                "c_BF_lower_from_c_OR": B.phi_or_to_bf(th, st["deg_i"], st["deg_j"]),
                "c_BF_upper_from_c_OR": B.psi_or_to_bf(th, st["deg_i"], st["deg_j"], st["C4"], st["matching"])}

    def _edgewise(self, x, name):
        a = np.asarray(x, dtype=float)
        M = self.num_edges
        if a.ndim == 0:
            return np.full(M, float(a))
        if a.shape == (M,):
            return a
        raise ValueError(f"{name} must be a scalar or have length {M}, got shape {a.shape}")

    # ---------------------------------------------------------- one call
    def compute_all(self, exact: bool = True, lly: bool = True,
                    exact_values: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Everything in one dict (keys of the preliminary code preserved).

        ``exact_values``: precomputed output of :meth:`exact` on all edges (e.g.
        loaded from a checkpoint); used instead of recomputing it.
        """
        st = self.stats()
        out: Dict[str, np.ndarray] = {"edges": self.edges.copy()}
        out.update(st)
        out["smax"] = st["matching"] / np.maximum(st["deg_i"], st["deg_j"])
        b = self.bounds()
        out.update(b)
        out["Theta_Const"], out["Theta_Slope"] = b["theta_const"], b["theta_slope"]
        if exact:
            ex = exact_values if exact_values is not None else \
                self.exact(("c_OR", "c_OR0", "c_LLY") if lly else ("c_OR", "c_OR0"))
            out.update(ex)
            out["prior_upper"] = np.minimum(ex["tlw_dual_lazy"], b["zero_cost_upper"])
            br = self.bounds_from_OR(ex["c_OR"])
            out["c_BF_lower_from_c_OR"] = br["c_BF_lower_from_c_OR"]
            out["c_BF_upper_from_c_OR"] = br["c_BF_upper_from_c_OR"]
        out["c_OR_lower_from_c_BF"] = b["bf_lower"]
        out["c_OR_upper_from_c_BF"] = b["bf_upper"]
        return out

    def edge_table(self, exact: bool = True, exact_values: Optional[Dict[str, np.ndarray]] = None):
        """Per-edge pandas DataFrame (columns compatible with the figure code)."""
        import pandas as pd

        r = self.compute_all(exact=exact, exact_values=exact_values)
        cols = {
            "u": r["edges"][:, 0], "v": r["edges"][:, 1],
            "deg_i": r["deg_i"].astype(int), "deg_j": r["deg_j"].astype(int),
            "triangle": r["triangle"].astype(int), "Xi": r["Xi"].astype(int),
            "sho_max": r["sho_max"], "C4": r["C4"], "matching": r["matching"].astype(int),
            "c_BF": r["c_BF"],
            "Theta_const": r["theta_const"], "Theta_slope": r["theta_slope"], "Theta_at_t": r["theta_at_t"],
            "env_upper": r["envelope_upper"],
            "cOR_lower_from_cBF": r["bf_lower"], "cOR_upper_from_cBF": r["bf_upper"],
            "cOR_lower_from_cBF_preliminary": r["bf_lower_preliminary"],
            "sharpened_lower": r["sharpened_lower"], "sharpened_upper": r["sharpened_upper"],
            "matching_lower_lazy": r["matching_lower_lazy"],
            "bf_deficit_A": r["bf_deficit_A"], "bf_deficit_B": r["bf_deficit_B"],
        }
        if exact:
            cols.update({
                "c_OR": r["c_OR"], "c_OR0": r["c_OR0"],
                "cBF_lower_from_cOR": r["c_BF_lower_from_c_OR"],
                "cBF_upper_from_cOR": r["c_BF_upper_from_c_OR"],
                "tlw_dual_lazy": r["tlw_dual_lazy"], "prior_upper": r["prior_upper"],
            })
            if "c_LLY" in r:
                cols["c_LLY"] = r["c_LLY"]
        return pd.DataFrame(cols)


def _as_graph(obj) -> Graph:
    if isinstance(obj, Graph):
        return obj
    if hasattr(obj, "edge_index") and hasattr(obj, "num_nodes"):  # PyG Data
        return Graph.from_edge_index(obj.edge_index, int(obj.num_nodes))
    if hasattr(obj, "nodes") and hasattr(obj, "edges") and callable(getattr(obj, "edges")):
        n = obj.number_of_nodes()
        if set(obj.nodes()) != set(range(n)):
            raise ValueError("networkx graph must be labeled 0..n-1; relabel with "
                             "nx.convert_node_labels_to_integers or use Graph.from_networkx")
        return Graph.from_edges(np.array(list(obj.edges()), dtype=np.int64).reshape(-1, 2), n)
    return Graph.from_edges(obj)
