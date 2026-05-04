"""Prototype: NLS-aware Hessian computation via jaxpr-level structure detection.

This package is a research prototype on the
`claude/hessian-symmetry-analysis-Fal50` feature branch.  It demonstrates
detecting nonlinear-least-squares structure in a JAX-traced objective and
splitting the Hessian into

    H(x)  =  2c · J(x)^T J(x)  +  2c · Σ_i r_i(x) ∇² r_i(x)

so that downstream tooling can route each branch through specialised
machinery (sparse Jacobian propagation for JᵀJ, pair-trim walker for the
residual correction, or the linear-residual fast path that skips the
correction entirely).

Modules:
    detect    — pure jaxpr inspection; returns NLSStructure or None.
    extract   — sub-jaxpr extraction utilities.
    split     — JᵀJ and residual-correction branches; reference impls.
    api       — top-level entry point with graceful non-NLS fallback.
    tests     — correctness + primitive-count diagnostics.

The split branches are overridable: pass `jtj_fn` / `correction_fn` to
`nls_hessian` to substitute pair-trim or sparse-Jacobian implementations
without modifying this package.
"""
from api import nls_hessian, is_nls
from detect import NLSStructure, detect_nls, detect_from_callable
from extract import residual_subjaxpr, as_callable
from split import (jtj_branch, residual_correction_branch, split_hessian,
                   is_linear_residual)
from walker import edge_push_correction, FORWARD_RULES, BACKWARD_RULES

__all__ = [
    "BACKWARD_RULES",
    "FORWARD_RULES",
    "NLSStructure",
    "as_callable",
    "detect_from_callable",
    "detect_nls",
    "edge_push_correction",
    "is_linear_residual",
    "is_nls",
    "jtj_branch",
    "nls_hessian",
    "residual_correction_branch",
    "residual_subjaxpr",
    "split_hessian",
]
