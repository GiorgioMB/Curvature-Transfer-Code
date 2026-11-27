"""The triangle-quadrangle bound on the Ollivier-Ricci curvature of every edge (Theorem 5.3), the full range of
Theorem 5.2, screen-then-solve (Algorithm 1), BF, and the baselines of BM and TLW. Exact curvatures come from
orcut (faster-or-curvature)."""
from .baselines import bm_tlw, snap
from .full_range import lower_and_relaxed, upper_end
from .screen import screen_then_solve
from .tq_general import tq_general
from .tq_lazy import bf, bounds_from, csr_sorted, tq_lazy

__all__ = ["tq_lazy", "bounds_from", "bf", "csr_sorted", "tq_general", "lower_and_relaxed", "upper_end",
           "screen_then_solve", "bm_tlw", "snap"]
__version__ = "2.0.0"
