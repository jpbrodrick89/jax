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
from extract import residual_subjaxpr
from split import split_hessian, is_linear_residual
from walker import edge_push_correction


def _walker_correction_fn_factory(closed_jaxpr, residual_var):
    """Wrap edge_push_correction into the (r_fn, x) -> H signature."""
    sub = residual_subjaxpr(closed_jaxpr, residual_var)
    def correction_fn(r_fn, x):
        r_value = r_fn(x)
        return edge_push_correction(sub, x, r_value)
    return correction_fn


def nls_hessian(f: Callable, x,
                jtj_fn=None, correction_fn=None,
                use_walker: bool = False):
    """Compute Hessian of f at x, exploiting NLS structure if present.

    Returns the full dense Hessian (a square matrix of shape (n, n)
    where n is the number of elements in x).

    For linear-residual problems the correction term is identically zero
    and is skipped (yields exactly H = 2c · JᵀJ).

    Pass `jtj_fn` and/or `correction_fn` to substitute custom branch
    implementations (e.g. a sparse-Jacobian propagator and a pair-trim
    walker).  Defaults are reference implementations via stock JAX.

    If `use_walker=True` and `correction_fn` is not given, the prototype
    edge-pushing walker (from `walker.py`) is used for the correction
    branch.  The walker is closer to the analytical optimum: a single
    forward + single backward sweep through the residual sub-jaxpr,
    accumulating per-primitive symmetric Hessian contributions.  It
    requires the residual jaxpr's primitives to be in the walker's
    rule registry; it raises NotImplementedError otherwise.
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
        import jax.numpy as jnp
        kwargs['correction_fn'] = lambda r_fn, x: jnp.zeros((x.size, x.size), x.dtype)
    elif use_walker:
        kwargs['correction_fn'] = _walker_correction_fn_factory(
            info.closed_jaxpr, info.residual_var)

    return split_hessian(info, x, **kwargs)


def is_nls(f: Callable, x) -> bool:
    """Return True iff `f` matches one of the NLS patterns at this argument."""
    return detect_from_callable(f, x) is not None
