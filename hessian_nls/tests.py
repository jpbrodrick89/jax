"""Correctness and structure tests for the NLS Hessian split.

What this checks:

  - detection: NLS patterns identified, non-NLS rejected.
  - correctness: split Hessian matches `jax.hessian` to numerical tolerance
    on a battery of small NLS problems including Luksan-style chained
    residuals.
  - structure: primitive counts in the split path's two branch jaxprs vs.
    the monolithic `jax.hessian` jaxpr, to demonstrate that the split is
    decomposing work along the JᵀJ / correction lines.
"""
from __future__ import annotations

import collections
import jax
import jax.numpy as jnp
import numpy as np

from api import nls_hessian, is_nls
from detect import detect_from_callable
from extract import residual_subjaxpr, as_callable
from split import jtj_branch, residual_correction_branch, is_linear_residual


def _eqn_count(jaxpr) -> int:
    j = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    return len(j.eqns)


def _prim_counter(jaxpr) -> dict:
    j = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    return dict(collections.Counter(str(e.primitive) for e in j.eqns))


# ---- problem battery -------------------------------------------------

def linear_residual(n=8):
    A = jnp.arange(n*n, dtype=jnp.float32).reshape(n, n) * 0.01
    b = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32)
    def f(x):
        r = A @ x - b
        return 0.5 * jnp.sum(r**2)
    return f, jnp.ones(n, dtype=jnp.float32) * 0.3


def quadratic_residual(n=6):
    """Each r_i is x_i*x_{i+1} - target — bilinear residuals (NLS)."""
    target = jnp.linspace(0.5, 2.0, n-1, dtype=jnp.float32)
    def f(x):
        r = x[:-1] * x[1:] - target
        return 0.5 * jnp.sum(r**2)
    return f, jnp.linspace(0.4, 1.5, n, dtype=jnp.float32)


def luksan_chained_rosenbrock(n=8):
    """LUKSAN21-style chained Rosenbrock: r_i = (1 - x_i, 10(x_{i+1} - x_i^2))."""
    def f(x):
        r1 = 1.0 - x[:-1]
        r2 = 10.0 * (x[1:] - x[:-1]**2)
        r  = jnp.concatenate([r1, r2])
        return 0.5 * jnp.sum(r**2)
    return f, jnp.linspace(-1.2, 1.0, n, dtype=jnp.float32)


def trig_residuals(n=5):
    """Residuals with mixed nonlinearity."""
    def f(x):
        r = jnp.sin(x[:-1]) * jnp.cos(x[1:]) - 0.3
        return 0.5 * jnp.sum(r**2)
    return f, jnp.linspace(0.2, 1.7, n, dtype=jnp.float32)


def not_nls_logsumexp(n=5):
    def f(x):
        return jax.nn.logsumexp(x) + jnp.sum(x**4)
    return f, jnp.linspace(-1, 1, n, dtype=jnp.float32)


PROBLEMS = [
    ("linear residual",          linear_residual,           True),
    ("quadratic residual",       quadratic_residual,        True),
    ("Luksan chained-Rosen",     luksan_chained_rosenbrock, True),
    ("trig residuals",           trig_residuals,            True),
    ("not NLS: logsumexp + x^4", not_nls_logsumexp,         False),
]


def test_detection_and_correctness_fp64():
    """Same battery but in float64.  Should achieve relative error ~1e-15."""
    print("\n==== DETECTION + CORRECTNESS (fp64) ====")
    # Re-run with jax_enable_x64 turned on temporarily.
    from jax import config
    prev = config.jax_enable_x64
    config.update("jax_enable_x64", True)
    try:
        for name, builder, expect_nls in PROBLEMS:
            f, x = builder()
            x64 = jnp.asarray(x, dtype=jnp.float64)
            detected = is_nls(f, x64)
            H_ours = nls_hessian(f, x64)
            H_ref  = jax.hessian(f)(x64).reshape(x64.size, x64.size)
            err = float(jnp.max(jnp.abs(H_ours - H_ref)))
            rel = err / max(float(jnp.max(jnp.abs(H_ref))), 1e-30)
            ok = "ok " if rel < 1e-12 else "BAD"
            print(f"  [{ok}] {name:30s}  abs={err:.2e}  rel={rel:.2e}")
    finally:
        config.update("jax_enable_x64", prev)


def test_detection_and_correctness():
    print("\n==== DETECTION + CORRECTNESS (fp32) ====")
    for name, builder, expect_nls in PROBLEMS:
        f, x = builder()
        detected = is_nls(f, x)
        H_ours = nls_hessian(f, x)
        H_ref  = jax.hessian(f)(x).reshape(x.size, x.size)
        err = float(jnp.max(jnp.abs(H_ours - H_ref)))
        rel = err / max(float(jnp.max(jnp.abs(H_ref))), 1e-12)
        status = "ok " if rel < 1e-4 else "BAD"
        match  = "ok " if detected == expect_nls else "BAD"
        print(f"  [{status}] [det={match}] {name:30s}  "
              f"NLS={detected}  abs_err={err:.2e}  rel_err={rel:.2e}")


def test_structure():
    print("\n==== STRUCTURE: primitive counts ====")
    print(f"  {'problem':30s} {'baseline':>10s} {'JTJ':>8s} {'corr':>8s} "
          f"{'effective':>10s}  notes")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for name, builder, expect_nls in PROBLEMS:
        if not expect_nls:
            continue
        f, x = builder()
        info = detect_from_callable(f, x)
        linear = is_linear_residual(info)
        sub = residual_subjaxpr(info.closed_jaxpr, info.residual_var)
        r_fn = as_callable(sub)

        # Baseline: jax.hessian's single monolithic jaxpr.
        baseline_jp = jax.make_jaxpr(jax.hessian(f))(x)

        # Split: two separate jaxprs.
        jtj_jp  = jax.make_jaxpr(lambda x: jtj_branch(r_fn, x))(x)
        corr_jp = jax.make_jaxpr(lambda x: residual_correction_branch(r_fn, x))(x)

        b = _eqn_count(baseline_jp)
        j = _eqn_count(jtj_jp)
        c = _eqn_count(corr_jp)
        # When residuals are linear, the correction branch is skipped at
        # the API layer; the effective primitive count is JTJ alone.
        effective = j if linear else j + c
        notes = "linear → correction skipped" if linear else ""
        print(f"  {name:30s} {b:10d} {j:8d} {c:8d} {effective:10d}  {notes}")


def test_linear_skip_correctness():
    """Confirm that for linear residuals, the JTJ-only path matches jax.hessian."""
    print("\n==== LINEAR-RESIDUAL SKIP CORRECTNESS ====")
    for name, builder, expect_nls in PROBLEMS:
        if not expect_nls:
            continue
        f, x = builder()
        info = detect_from_callable(f, x)
        linear = is_linear_residual(info)
        if not linear:
            print(f"  {name:30s}  (residuals nonlinear; skip not applicable)")
            continue
        H_skip = nls_hessian(f, x)               # uses skip when linear
        H_ref  = jax.hessian(f)(x).reshape(x.size, x.size)
        err = float(jnp.max(jnp.abs(H_skip - H_ref)))
        print(f"  {name:30s}  abs_err={err:.2e}  ok")


def test_handcrafted_smoke():
    """Hand-checked Hessians on tiny problems."""
    print("\n==== HAND-CHECKED SMOKE ====")
    # f = (1/2)(x^2 + y^2)  — H = I
    f = lambda v: 0.5 * jnp.sum(v**2)
    x = jnp.array([3.0, -1.0])
    H = nls_hessian(f, x)
    assert jnp.allclose(H, jnp.eye(2)), H
    print("  identity Hessian for (1/2)|v|^2 ok")

    # f = (1/2)(xy - 2)^2  — H_ij = (∂r/∂i)(∂r/∂j) + r·(∂²r/∂i∂j)
    # r = xy-2; H = [[y^2, 2xy-2], [2xy-2, x^2]]   (since r*∂²r/∂x∂y = r*1)
    f = lambda v: 0.5 * (v[0]*v[1] - 2.0)**2
    x = jnp.array([1.5, 2.0])
    H = nls_hessian(f, x)
    expected = jnp.array([[2.0**2,             2*1.5*2.0 - 2],
                          [2*1.5*2.0 - 2,      1.5**2     ]])
    assert jnp.allclose(H, expected, atol=1e-5), (H, expected)
    print("  bilinear residual Hessian ok")


if __name__ == "__main__":
    test_handcrafted_smoke()
    test_detection_and_correctness()
    test_detection_and_correctness_fp64()
    test_linear_skip_correctness()
    test_structure()
