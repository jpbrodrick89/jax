"""Hand-rolled NLS Hessian routines that take a residual function directly.

These are the "AD best case": what any user could write by hand if they
already have `r(x)` exposed, without going through the jaxpr-detection
pipeline.  Used as a comparison point in benchmarks — they bound below
how fast the auto-detected `nls_hessian` could ever go on a given
problem with stock JAX components, since they share the same underlying
AD math but skip detection / sub-jaxpr extraction overhead.

Three variants:

    manual_jtj           — Gauss-Newton approximation only.  Cheap, the
                           thing many NLS solvers actually use (LM,
                           trust-region with GN inner solve).  Exact
                           when residuals are linear, otherwise an
                           approximation.

    manual_split         — Full Hessian via the JᵀJ + correction split.
                           Same math as `nls_hessian` but skips
                           detection.  Should be the realistic AD lower
                           bound for the full Hessian.

    manual_hessian_phi   — Single-call alternative: skips the JᵀJ split
                           and just does `jax.hessian` on
                           φ(y) = ½‖r(y)‖² (i.e. the original f).
                           Equivalent to plain `jax.hessian(f)` but
                           expressed through r — useful sanity check.

All three take `r_fn` returning a 1-D residual vector and a 1-D `x`.
The result is a dense (n, n) Hessian; use a sparse backend by replacing
the matmul / accumulation steps in your own copy.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def _pick_jac(r_fn: Callable, n: int, m: int):
    """Forward mode when n ≤ m, reverse otherwise."""
    return jax.jacfwd(r_fn) if n <= m else jax.jacrev(r_fn)


def manual_jtj(r_fn: Callable, x, half_factor: float = 0.5):
    """Gauss-Newton approximation:  H ≈ 2c · JᵀJ.

    Exact for linear residuals; for nonlinear residuals omits the
    `Σᵢ rᵢ ∇²rᵢ` correction term (which vanishes near the solution).
    """
    n = int(jnp.asarray(x).size)
    r_val = r_fn(x)
    m = int(r_val.size)
    J = _pick_jac(r_fn, n, m)(x)
    J = J.reshape(m, n)
    return (2.0 * half_factor) * (J.T @ J)


def manual_split(r_fn: Callable, x, half_factor: float = 0.5):
    """Full NLS Hessian via the JᵀJ + correction split.

    Best-case AD implementation given r_fn directly:
      JᵀJ:        one Jacobian assembly + matmul.
      correction: jax.hessian on  φ(y) = ⟨ stop_grad(r(x)), r(y) ⟩.
    """
    n = int(jnp.asarray(x).size)
    r_val = r_fn(x)
    m = int(r_val.size)

    J = _pick_jac(r_fn, n, m)(x).reshape(m, n)
    JTJ = J.T @ J

    r_const = jax.lax.stop_gradient(r_val)
    phi = lambda y: jnp.dot(r_const, r_fn(y))
    correction = jax.hessian(phi)(x).reshape(n, n)

    return (2.0 * half_factor) * (JTJ + correction)


def manual_hessian_phi(r_fn: Callable, x, half_factor: float = 0.5):
    """Direct `jax.hessian` on f(y) = c‖r(y)‖² — no split, just AD.

    This is mathematically identical to `jax.hessian(f)(x)` for the
    corresponding `f`, but reuses an existing `r_fn`.  Included as a
    sanity-check against the split approach.
    """
    n = int(jnp.asarray(x).size)
    f = lambda y: half_factor * jnp.sum(r_fn(y)**2)
    return jax.hessian(f)(x).reshape(n, n)
