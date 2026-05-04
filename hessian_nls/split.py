"""Compute the Hessian of a sum-of-squares objective via the structural split.

Given an NLS structure `f(x) = c · Σ r_i(x)²` with c = half_factor:

    H(x) = 2c · J(x)^T J(x)  +  2c · Σ_i r_i(x) ∇² r_i(x)
           └──────────────┘    └────────────────────────┘
              JᵀJ branch         residual-correction branch

This module computes both branches and returns their sum.

Architectural notes
-------------------
The two branches are intentionally implemented through different code
paths so that downstream users can substitute their own machinery:

- JᵀJ branch: forms J as a sparse/dense Jacobian (one AD pass), then
  contracts.  In a pair-trim-aware tooling this is replaced with a
  sparse-Jacobian-propagation walk over the residual sub-jaxpr.

- Residual-correction branch: builds the linear-in-r̄ scalar function
  φ(x) = ⟨r̄, r(x)⟩ with r̄ frozen at r(x₀), and takes its Hessian.
  This is a smaller Hessian problem (no squaring layer) and is the
  natural target for a pair-trim walker.  In this prototype we just
  call `jax.hessian` on φ; a real implementation would call the user's
  pair-trim walker here.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from detect import NLSStructure
from extract import residual_subjaxpr, as_callable


def _flatten_argument(x):
    """Treat x as a flat 1D array for matrix-form J^T J operations.

    For the prototype we restrict to a single ndarray argument.  Multi-
    argument flattening (PyTrees) is a straightforward extension.
    """
    return jnp.asarray(x).reshape(-1)


def jtj_branch(r_fn: Callable, x):
    """Compute J(x)^T J(x) by forming J explicitly with `jax.jacrev`.

    For a residual r: R^n -> R^m, jacrev costs m backward passes (good
    for m << n) and jacfwd costs n forward passes (good for n << m).
    A production implementation would pick whichever is smaller and a
    pair-trim-aware version would propagate sparse Jacobians directly.
    """
    J = jax.jacrev(r_fn)(x)
    # J has shape (m, *x.shape).  Flatten the trailing dims.
    J_mat = J.reshape(J.shape[0], -1)
    return J_mat.T @ J_mat


def residual_correction_branch(r_fn: Callable, x):
    """Compute Σ_i r_i(x) ∇² r_i(x) at x.

    Implemented via:
        φ(x') = ⟨ stop_grad(r(x)), r(x') ⟩
        correction = ∇²φ(x)
    Then ∇²φ = Σ_i r_i ∇²r_i because the stop-gradded prefactor freezes
    the cotangent at r_i.

    A pair-trim-aware tooling would substitute its own walker here:
    the input to the walker is the residual sub-jaxpr plus the
    cotangent-vector r(x); the output is the correction matrix.
    """
    r_at_x = jax.lax.stop_gradient(r_fn(x))
    phi = lambda y: jnp.dot(r_at_x, r_fn(y))
    return jax.hessian(phi)(x)


def split_hessian(info: NLSStructure, x,
                  jtj_fn: Callable = jtj_branch,
                  correction_fn: Callable = residual_correction_branch):
    """Compute Hessian of f(x) = c · Σ r_i(x)² using the JᵀJ + correction split.

    Both branches are overridable so downstream users can swap in their
    own machinery without modifying this module:

      - `jtj_fn(r_fn, x) -> (n, n) ndarray`         e.g. sparse-J propagation.
      - `correction_fn(r_fn, x) -> (n, n) ndarray`  e.g. pair-trim walker.

    The defaults (`jtj_branch`, `residual_correction_branch`) are
    self-contained reference implementations using stock JAX primitives.
    """
    sub = residual_subjaxpr(info.closed_jaxpr, info.residual_var)
    r_fn = as_callable(sub)

    JTJ = jtj_fn(r_fn, x)
    corr = correction_fn(r_fn, x)

    scale = 2.0 * info.half_factor
    n = JTJ.shape[0]
    return scale * (JTJ + corr.reshape(n, n))


def is_linear_residual(info: NLSStructure) -> bool:
    """Return True iff the residual sub-jaxpr is linear in the function inputs.

    Linearity here is w.r.t. the *invars* of the closed jaxpr — closed-over
    constants (constvars and Literals) are treated as constants and are
    *not* counted toward bilinearity.  When this returns True, the residual
    correction term Σ r_i ∇²r_i is exactly zero and the correction branch
    can be skipped.
    """
    from jax.extend.core import Literal
    sub = residual_subjaxpr(info.closed_jaxpr, info.residual_var)
    sj = sub.jaxpr

    # Track "x-dependence": vars derived (transitively) from invars.
    x_dep = set(sj.invars)

    LINEAR_PRIMS = {
        'add', 'sub', 'neg', 'convert_element_type',
        'reshape', 'broadcast_in_dim', 'squeeze',
        'transpose', 'slice', 'dynamic_slice',
        'pad', 'concatenate', 'reduce_sum',
        'select_n',
    }

    def is_x_dep(v):
        return isinstance(v, type(sj.invars[0])) and v in x_dep

    for eqn in sj.eqns:
        name = eqn.primitive.name
        x_dep_inputs = [v for v in eqn.invars
                        if not isinstance(v, Literal) and v in x_dep]
        nonlinear_in_x = False

        if name in LINEAR_PRIMS:
            pass
        elif name in ('mul', 'div', 'dot_general'):
            # Linear in x iff at most one of the operands is x-dependent.
            if len(x_dep_inputs) > 1:
                nonlinear_in_x = True
        else:
            # Any elementwise nonlinear (sin, cos, exp, log, integer_pow, ...)
            # applied to an x-dependent operand is nonlinear in x.
            if x_dep_inputs:
                nonlinear_in_x = True

        if nonlinear_in_x:
            return False

        # Outputs of a linear-in-x equation are x-dependent iff any input is.
        if x_dep_inputs:
            for ov in eqn.outvars:
                x_dep.add(ov)

    return True
