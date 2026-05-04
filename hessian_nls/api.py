"""Top-level entry point: NLS-aware Hessian with graceful fallback.

Usage:
    H = nls_hessian(f, x)

If `f` has the shape `c · Σ r_i(x)²` (any of the patterns documented in
`detect.py`), `nls_hessian` routes the work through the JᵀJ + residual-
correction split.  Otherwise it falls back to `jax.hessian(f)(x)`.

The fallback is deliberate: this module is concerned with NLS structure
only, and any non-NLS objective is exactly what existing Hessian
machinery already handles well.
"""
from __future__ import annotations

from typing import Callable

import jax

from detect import detect_from_callable
from split import split_hessian, is_linear_residual


def nls_hessian(f: Callable, x,
                jtj_fn=None, correction_fn=None):
    """Compute Hessian of f at x, exploiting NLS structure if present.

    Returns the full dense Hessian (a square matrix of shape (n, n)
    where n is the number of elements in x).

    For linear-residual problems the correction term is identically zero
    and is skipped (yields exactly H = 2c · JᵀJ).

    Pass `jtj_fn` and/or `correction_fn` to substitute custom branch
    implementations (e.g. a sparse-Jacobian propagator and a pair-trim
    walker).  Defaults are reference implementations via stock JAX.
    """
    info = detect_from_callable(f, x)
    if info is None:
        return jax.hessian(f)(x).reshape(x.size, x.size)

    kwargs = {}
    if jtj_fn is not None:
        kwargs['jtj_fn'] = jtj_fn
    if correction_fn is not None:
        kwargs['correction_fn'] = correction_fn
    elif is_linear_residual(info):
        # Skip correction branch entirely: it is identically zero.
        import jax.numpy as jnp
        kwargs['correction_fn'] = lambda r_fn, x: jnp.zeros((x.size, x.size), x.dtype)

    return split_hessian(info, x, **kwargs)


def is_nls(f: Callable, x) -> bool:
    """Return True iff `f` matches one of the NLS patterns at this argument."""
    return detect_from_callable(f, x) is not None
