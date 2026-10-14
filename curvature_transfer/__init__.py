"""curvature_transfer: certified bounds between Balanced Forman and
Ollivier--Ricci curvature, and exact OR curvature by min-cut.

Core dependencies are numpy, scipy, and numba; torch / torch_geometric are
only needed for :mod:`curvature_transfer.pyg`.
"""
from .graph import Graph
from .engine import CurvatureEngine
from . import bounds
from .screening import certify_threshold, certify_bottom_k, certify_then_solve

__all__ = ["Graph", "CurvatureEngine", "bounds",
           "certify_threshold", "certify_bottom_k", "certify_then_solve"]
__version__ = "1.0.0"
