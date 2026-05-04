"""Detect whether a closed jaxpr computes a sum-of-squares objective.

Pure jaxpr inspection — no math, no derivative work, no jax.grad.
Returns a small dataclass identifying the residual variable and any
multiplicative scaling factor on the outer sum.

Detection patterns recognised (the output equation chain, walking back
from the unique scalar output):

  out = reduce_sum(s)                          half_factor = 1.0
  out = mul(c,   reduce_sum(s))                half_factor = c   (any literal)

with s produced by:
  s = integer_pow(r, 2)                        residual = r
  s = mul(r, r)  with the two operands the same SSA var
                                               residual = r

OR, single-equation form via dot_general contracted over all axes:
  out = dot_general(r, r)                      residual = r,  half_factor = 1.0
  out = mul(c, dot_general(r, r))              residual = r,  half_factor = c

If any link of the chain fails, returns None and the caller should fall
back to the generic Hessian path.

Note: writing `r*r` or `jnp.dot(r, r)` *without* binding `r` to a name
produces two structurally-distinct SSA vars in the jaxpr (because JAX
does not CSE the duplicated expression).  The detector requires the
operand-equal form, which arises naturally when the residual is bound
to a name before being squared.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import jax
from jax.extend.core import ClosedJaxpr, Jaxpr, Var, Literal


@dataclasses.dataclass(frozen=True)
class NLSStructure:
    """Identification of NLS structure inside a closed jaxpr.

    Attributes:
        closed_jaxpr:  the original closed jaxpr of f(x).
        residual_var:  SSA Var inside closed_jaxpr.jaxpr whose value is r(x).
        half_factor:   scalar prefactor on the outer reduction
                       (0.5 for standard NLS f = ½‖r‖², 1.0 for f = ‖r‖²).
    """
    closed_jaxpr: ClosedJaxpr
    residual_var: Var
    half_factor: float


def _eqn_defining(jaxpr: Jaxpr, var: Var):
    """Return the equation whose outvars contain `var`, or None."""
    for eqn in jaxpr.eqns:
        if any(v is var for v in eqn.outvars):
            return eqn
    return None


def _as_literal_float(v) -> Optional[float]:
    """If v is a scalar Literal with a float-coercible value, return it."""
    if not isinstance(v, Literal):
        return None
    try:
        val = v.val
    except AttributeError:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _try_match_dot_self(eqn) -> Optional[Var]:
    """Match `dot_general(r, r)` contracting all axes (scalar output).

    Returns r if the equation is operand-equal `dot_general` whose output
    is a scalar (i.e. all axes of r are contracted), else None.
    """
    if eqn.primitive.name != 'dot_general':
        return None
    if len(eqn.invars) != 2:
        return None
    a, b = eqn.invars
    if not (isinstance(a, Var) and a is b):
        return None
    if eqn.outvars[0].aval.shape != ():
        return None
    return a


def _try_match_sum_of_square(jaxpr: Jaxpr, eqn) -> Optional[Var]:
    """Match `reduce_sum(integer_pow(r, 2))` or `reduce_sum(mul(r, r))`."""
    if eqn.primitive.name != 'reduce_sum':
        return None
    if len(eqn.invars) != 1:
        return None
    sq_var = eqn.invars[0]
    if not isinstance(sq_var, Var):
        return None
    sq_eqn = _eqn_defining(jaxpr, sq_var)
    if sq_eqn is None:
        return None

    if (sq_eqn.primitive.name == 'integer_pow'
            and sq_eqn.params.get('y') == 2
            and len(sq_eqn.invars) == 1
            and isinstance(sq_eqn.invars[0], Var)):
        return sq_eqn.invars[0]

    if (sq_eqn.primitive.name == 'mul'
            and len(sq_eqn.invars) == 2
            and isinstance(sq_eqn.invars[0], Var)
            and sq_eqn.invars[0] is sq_eqn.invars[1]):
        return sq_eqn.invars[0]

    return None


def detect_nls(closed_jaxpr: ClosedJaxpr) -> Optional[NLSStructure]:
    """Detect NLS structure in a closed jaxpr.  Returns None if not NLS."""
    jaxpr = closed_jaxpr.jaxpr

    # Single scalar output required.
    if len(jaxpr.outvars) != 1:
        return None
    out_var = jaxpr.outvars[0]
    if not isinstance(out_var, Var):
        return None
    if out_var.aval.shape != ():
        return None

    out_eqn = _eqn_defining(jaxpr, out_var)
    if out_eqn is None:
        return None

    # Peel off optional outer mul(scalar, ...).
    half_factor = 1.0
    inner_eqn = out_eqn
    if out_eqn.primitive.name == 'mul':
        if len(out_eqn.invars) != 2:
            return None
        a, b = out_eqn.invars
        a_lit, b_lit = _as_literal_float(a), _as_literal_float(b)
        if a_lit is not None and isinstance(b, Var):
            half_factor, peel_var = a_lit, b
        elif b_lit is not None and isinstance(a, Var):
            half_factor, peel_var = b_lit, a
        else:
            # mul of two non-literal values — not the outer scalar prefactor.
            return None
        inner_eqn = _eqn_defining(jaxpr, peel_var)
        if inner_eqn is None:
            return None

    residual_var = _try_match_dot_self(inner_eqn)
    if residual_var is None:
        residual_var = _try_match_sum_of_square(jaxpr, inner_eqn)
    if residual_var is None:
        return None

    return NLSStructure(
        closed_jaxpr=closed_jaxpr,
        residual_var=residual_var,
        half_factor=half_factor,
    )


def detect_from_callable(f, *args) -> Optional[NLSStructure]:
    """Convenience wrapper: trace f(*args) to a jaxpr, then detect."""
    return detect_nls(jax.make_jaxpr(f)(*args))
