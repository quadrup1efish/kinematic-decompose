"""Compatibility imports for stellar assembly; use the analysis package in new code."""

from ..analysis.stellar_assembly import cal_in_ex_situ, compute_stellar_assembly

__all__ = ["cal_in_ex_situ", "compute_stellar_assembly"]
