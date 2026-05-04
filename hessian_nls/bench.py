"""Wall-clock benchmarks: split path vs jax.hessian baseline.

Run with float64 for stable timings.  Both paths are JIT-compiled with
identical setup before timing; warmup runs are excluded from the median.

Caveats:
  - CPU timings.  GPU/TPU numbers will differ (especially on the JᵀJ
    branch, where matmul is the dominant op and benefits from XLA fusion).
  - The correction branch uses stock `jax.hessian(φ)` here.  In practice
    you'd plug in a pair-trim walker and that branch would shrink.
  - We time the per-call cost after compilation; compile time is shown
    separately in a small column.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from api import nls_hessian, is_nls
from detect import detect_from_callable
from extract import residual_subjaxpr, as_callable
from manual import manual_jtj, manual_split, manual_hessian_phi
from split import jtj_branch, residual_correction_branch, is_linear_residual


# ---- problem builders -----------------------------------------------

def linear_residual(n):
    rng = jax.random.PRNGKey(0)
    A = jax.random.normal(rng, (n, n), dtype=jnp.float64) * (1.0 / n**0.5)
    b = jax.random.normal(jax.random.PRNGKey(1), (n,), dtype=jnp.float64)
    def r(x): return A @ x - b
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.ones(n, dtype=jnp.float64) * 0.3


def quadratic_residual(n):
    target = jnp.linspace(0.5, 2.0, n-1, dtype=jnp.float64)
    def r(x): return x[:-1] * x[1:] - target
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.linspace(0.4, 1.5, n, dtype=jnp.float64)


def luksan_chained_rosenbrock(n):
    def r(x):
        return jnp.concatenate([1.0 - x[:-1], 10.0 * (x[1:] - x[:-1]**2)])
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.linspace(-1.2, 1.0, n, dtype=jnp.float64)


def trig_residuals(n):
    def r(x): return jnp.sin(x[:-1]) * jnp.cos(x[1:]) - 0.3
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.linspace(0.2, 1.7, n, dtype=jnp.float64)


def overdetermined_linear(n):
    m = 8 * n
    rng = jax.random.PRNGKey(0)
    A = jax.random.normal(rng, (m, n), dtype=jnp.float64) * (1.0 / m**0.5)
    b = jax.random.normal(jax.random.PRNGKey(1), (m,), dtype=jnp.float64)
    def r(x): return A @ x - b
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.ones(n, dtype=jnp.float64) * 0.3


def overdetermined_nonlinear(n):
    m = 4 * n
    rng = jax.random.PRNGKey(0)
    A = jax.random.normal(rng, (m, n), dtype=jnp.float64) * (1.0 / n**0.5)
    b = jax.random.normal(jax.random.PRNGKey(1), (m,), dtype=jnp.float64)
    def r(x): return jnp.tanh(A @ x) - b
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.zeros(n, dtype=jnp.float64) + 0.1


def underdetermined_linear(n):
    m = max(2, n // 10)
    rng = jax.random.PRNGKey(0)
    A = jax.random.normal(rng, (m, n), dtype=jnp.float64) * (1.0 / n**0.5)
    b = jax.random.normal(jax.random.PRNGKey(1), (m,), dtype=jnp.float64)
    def r(x): return A @ x - b
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.ones(n, dtype=jnp.float64) * 0.3


def underdetermined_nonlinear(n):
    m = max(2, n // 10)
    rng = jax.random.PRNGKey(0)
    A = jax.random.normal(rng, (m, n), dtype=jnp.float64) * (1.0 / n**0.5)
    b = jax.random.normal(jax.random.PRNGKey(1), (m,), dtype=jnp.float64)
    def r(x): return jnp.tanh(A @ x)
    def f(x): return 0.5 * jnp.sum(r(x)**2)
    return f, r, jnp.zeros(n, dtype=jnp.float64) + 0.1


PROBLEMS = [
    ("linear residual (m=n)",    linear_residual),
    ("quadratic residual",       quadratic_residual),
    ("Luksan chained-Rosen",     luksan_chained_rosenbrock),
    ("trig residuals",           trig_residuals),
    ("overdet. linear (m=8n)",   overdetermined_linear),
    ("overdet. tanh (m=4n)",     overdetermined_nonlinear),
    ("underdet. linear (m=n/10)", underdetermined_linear),
    ("underdet. tanh (m=n/10)",   underdetermined_nonlinear),
]


# ---- timing harness --------------------------------------------------

def _time_fn(fn, x, n_warmup=3, n_runs=20):
    # Warmup (compiles, primes caches).
    for _ in range(n_warmup):
        out = fn(x)
        jax.block_until_ready(out)
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(x)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _time_compile(fn_uncompiled, x):
    """Measure compile-time-on-first-call (rough)."""
    fn = jax.jit(fn_uncompiled)
    t0 = time.perf_counter()
    out = fn(x)
    jax.block_until_ready(out)
    return time.perf_counter() - t0, fn


def benchmark_problem(name: str, builder: Callable, sizes):
    print(f"\n===== {name} =====")
    print(f"  {'n':>5}  {'jax.hess':>10}  {'split':>10}  {'walker':>10}  "
          f"{'man.split':>10}  {'man.GN':>10}  "
          f"{'sp.split':>9}  {'sp.man':>9}  {'sp.GN':>9}  {'lin?':>5}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*5}")
    for n in sizes:
        f, r, x = builder(n)
        info = detect_from_callable(f, x)
        linear = is_linear_residual(info) if info is not None else False

        baseline    = jax.jit(jax.hessian(f))
        stock       = jax.jit(lambda x: nls_hessian(f, x))
        walker      = jax.jit(lambda x: nls_hessian(f, x, use_walker=True))
        man_split   = jax.jit(lambda x: manual_split(r, x))
        man_jtj     = jax.jit(lambda x: manual_jtj(r, x))   # GN approx

        # Correctness sanity (fp64).
        H_b = baseline(x);  jax.block_until_ready(H_b)
        H_s = stock(x);     jax.block_until_ready(H_s)
        H_m = man_split(x); jax.block_until_ready(H_m)
        ok_split = float(jnp.max(jnp.abs(H_b.reshape(n, n) - H_s))) < 1e-9
        ok_man   = float(jnp.max(jnp.abs(H_b.reshape(n, n) - H_m))) < 1e-9
        if not (ok_split and ok_man):
            print(f"  {n:>5}  CORRECTNESS FAILURE")
            continue

        try:
            H_w = walker(x); jax.block_until_ready(H_w)
            walker_ok = bool(jnp.max(jnp.abs(H_b.reshape(n, n) - H_w)) < 1e-9)
        except Exception:
            walker_ok = False

        t_b  = _time_fn(baseline, x)
        t_s  = _time_fn(stock, x)
        t_m  = _time_fn(man_split, x)
        t_gn = _time_fn(man_jtj, x)
        if walker_ok:
            t_w = _time_fn(walker, x)
            tw_str = f"{t_w*1e3:7.3f} ms"
        else:
            tw_str = "      n/a"
        sp_s  = t_b / t_s
        sp_m  = t_b / t_m
        sp_gn = t_b / t_gn

        lin = "yes" if linear else "no"
        print(f"  {n:>5}  {t_b*1e3:7.3f} ms  {t_s*1e3:7.3f} ms  {tw_str:>10}  "
              f"{t_m*1e3:7.3f} ms  {t_gn*1e3:7.3f} ms  "
              f"{sp_s:7.2f}x  {sp_m:7.2f}x  {sp_gn:7.2f}x  {lin:>5}")


def benchmark_branches(name: str, builder: Callable, n: int):
    """Timing breakdown of the two split branches at a fixed size."""
    print(f"\n----- branch timing: {name}, n={n} -----")
    f, r_fn, x = builder(n)
    info = detect_from_callable(f, x)
    if info is None:
        print("  (not NLS, skipped)")
        return
    sub = residual_subjaxpr(info.closed_jaxpr, info.residual_var)
    r_fn_extracted = as_callable(sub)

    from walker import edge_push_correction
    fns = {
        "jax.hessian (baseline)":   jax.jit(jax.hessian(f)),
        "manual_jtj (GN approx)":   jax.jit(lambda x: manual_jtj(r_fn, x)),
        "manual_split (best AD)":   jax.jit(lambda x: manual_split(r_fn, x)),
        "manual_hessian_phi":       jax.jit(lambda x: manual_hessian_phi(r_fn, x)),
        "nls_hessian stock":        jax.jit(lambda x: nls_hessian(f, x)),
        "nls_hessian walker":       jax.jit(lambda x: nls_hessian(f, x, use_walker=True)),
        "JTJ branch (extracted)":   jax.jit(lambda x: jtj_branch(r_fn_extracted, x)),
        "correction (stock)":       jax.jit(lambda x: residual_correction_branch(r_fn_extracted, x)),
        "correction (walker)":      jax.jit(lambda x: edge_push_correction(sub, x, r_fn_extracted(x))),
    }
    for label, fn in fns.items():
        t = _time_fn(fn, x)
        print(f"  {label:30s}  {t*1e3:9.3f} ms")


if __name__ == "__main__":
    SIZES = [10, 20, 50, 100, 200]

    for name, builder in PROBLEMS:
        benchmark_problem(name, builder, SIZES)

    # Detailed branch breakdown on representative problems
    benchmark_branches("Luksan chained-Rosen", luksan_chained_rosenbrock, n=100)
    benchmark_branches("linear residual",      linear_residual,           n=100)
    benchmark_branches("overdet. linear",      overdetermined_linear,     n=50)
    benchmark_branches("overdet. tanh",        overdetermined_nonlinear,  n=50)
    benchmark_branches("underdet. linear",     underdetermined_linear,    n=200)
    benchmark_branches("underdet. tanh",       underdetermined_nonlinear, n=200)
