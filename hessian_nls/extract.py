"""Extract a residual sub-jaxpr from a forward jaxpr.

Given a closed jaxpr J(x) and a Var v inside J that holds the residual
vector, build a new closed jaxpr R(x) whose output is v.  The new jaxpr
contains only the equations needed to compute v (transitively),
preserving J's invars and constvars.

This is pure jaxpr surgery — no AD, no math.
"""
from __future__ import annotations

from jax.extend.core import ClosedJaxpr, Jaxpr, Var, jaxpr_as_fun


def residual_subjaxpr(closed_jaxpr: ClosedJaxpr, residual_var: Var) -> ClosedJaxpr:
    """Return a closed jaxpr whose single output is `residual_var`.

    The returned jaxpr:
      - shares invars and constvars with `closed_jaxpr`,
      - contains exactly the equations from `closed_jaxpr` whose outputs
        are reachable backward from `residual_var`,
      - has `residual_var` as its sole outvar.
    """
    src = closed_jaxpr.jaxpr
    needed = {residual_var}
    kept = []
    # Walk equations in reverse; keep any equation that produces a needed var.
    for eqn in reversed(src.eqns):
        if any(v in needed for v in eqn.outvars):
            kept.append(eqn)
            for v in eqn.invars:
                if isinstance(v, Var):
                    needed.add(v)
    kept.reverse()

    new_jaxpr = Jaxpr(
        constvars=src.constvars,
        invars=src.invars,
        outvars=[residual_var],
        eqns=kept,
        effects=set(),
    )
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)


def as_callable(closed_jaxpr: ClosedJaxpr):
    """Wrap a ClosedJaxpr as a Python callable returning a single output."""
    fn = jaxpr_as_fun(closed_jaxpr)
    def call(*args):
        out = fn(*args)
        return out[0] if isinstance(out, list) else out
    return call
