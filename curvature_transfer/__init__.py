"""curvature_transfer: certified bounds between Balanced Forman and
Ollivier--Ricci curvature. Exact OR curvature comes from ``orcut`` (1.0.0).

Core dependencies are numpy, scipy, numba, and orcut.
"""
from .graph import Graph
from .engine import CurvatureEngine
from . import bounds
from . import intervals
from .screening import certify_threshold, certify_then_solve, screen

__all__ = ["Graph", "CurvatureEngine", "bounds", "intervals",
           "certify_threshold", "certify_then_solve", "screen"]
__version__ = "1.1.0"
