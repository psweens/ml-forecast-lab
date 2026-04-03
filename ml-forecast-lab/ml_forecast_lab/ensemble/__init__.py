"""
Ensemble engine for combining predictions from multiple forecast models.

Provides simple averaging, inverse-metric weighting, and stacking
(Ridge meta-learner) strategies for combining model predictions.
"""

from .engine import EnsembleEngine, EnsembleResult, EnsembleStrategy

__all__ = ["EnsembleEngine", "EnsembleResult", "EnsembleStrategy"]
