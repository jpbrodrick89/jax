"""DMN15102LS case study: where the 15-30x analytical speedup actually comes from.

Empirical finding (DMN15102LS, n=66, m=4643):

    operation                          time     factor
    ------------------------------------------------
    r(y)                               0.31 ms    1×
    J_analytic (closed form J)         0.33 ms    1×
    jacfwd(r)  [n=66 JVP passes]      31.5  ms  100×    <-- AD bottleneck
    H_analytic_full (closed form)      1.31 ms    4×
    jax.hessian(f)                    64.6  ms  208×    <-- 49× slower than analytic

The 49× speedup of `H_analytic_full` vs `jax.hessian` for DMN15102LS
breaks down as:

  1. **Jacobian via single batched op**, not jacfwd:
     `J_analytic` is ONE broadcast computation on (m, n_peaks). It re-uses
     the (4643, 33) Lorentzian intermediate that the residual already
     computes.  jacfwd reruns this intermediate 66 times — once per basis
     vector — because each JVP traces r with a different tangent.

  2. **Sparse correction structure**: ∇²r_i has only per-peak 2×2 blocks
     nonzero (∂²r/∂w_j ∂w_k = 0, ∂²r/∂w_j ∂width_k nonzero only j=k, etc.).
     `H_analytic_full` exploits this; jax.hessian materialises the full
     n×n correction matrix.

The "reduce_sum(dx*dx) → dot_general(Jᵀ, J)" rewrite the user asked about
is one symptom of (1): in the HVP jaxpr, the AD-generated chain expresses
`Σᵢ(Jᵢv)²` as a reduce_sum-over-elementwise-square.  Recognising it as
vᵀJᵀJv lets you precompute JᵀJ once.  But this is just a downstream view
of the same issue — the bigger problem is that AD computes J one column
at a time via per-basis-vector JVP passes, rather than as a batched
broadcast.  Once you compute J in one batched op, JᵀJ is one matmul and
the reduce_sum-pattern issue dissolves.

To automate this for arbitrary residuals, you'd need a jaxpr-level rewrite
that detects "residual = reduce(broadcast-elementwise(params))" patterns
and emits batched Jacobian computation directly.  That's a non-trivial
analysis but it's a single recognisable family covering most NLS test
problems whose residual is "evaluate a parametric model at m data points
and subtract observations".
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# These are problem-specific data loaders — the case-study object is the
# closed-form derivative formulas below, not the data.
def _load_data():
    from sif2jax.cutest._unconstrained_minimisation.dmn15102ls import _DMN15102LS_DATA
    return (jnp.asarray(_DMN15102LS_DATA["x_data"], jnp.float64),
            jnp.asarray(_DMN15102LS_DATA["y_data"], jnp.float64),
            jnp.asarray(_DMN15102LS_DATA["positions"], jnp.float64))


_X_DATA, _Y_DATA, _POSITIONS = _load_data()
_PI_INV = 1.0 / jnp.pi


def r_fn(y):
    """Residual: r_i(y) = Σ_k Lorentzian(x_i; w_k, width_k, pos_k) - y_i.

    Variables are interleaved: y[0::2] = weights w, y[1::2] = widths.
    """
    w  = y[0::2]; wd = y[1::2]
    x_e = _X_DATA[:, None]; p_e = _POSITIONS[None, :]
    D = (x_e - p_e)**2 + wd[None, :]**2
    lor = w[None, :] * _PI_INV * wd[None, :] / D
    return jnp.sum(lor, axis=1) - _Y_DATA


def J_analytic(y):
    """Closed-form Jacobian of r(y), in one batched broadcast op.

    Cost: O(m · n_peaks), same as evaluating r once.
    Compare with jacfwd(r)(y) which does n forward passes — for this
    problem (n=66) that's a 100× FLOP overhead.
    """
    w  = y[0::2]; wd = y[1::2]
    x_e = _X_DATA[:, None]; p_e = _POSITIONS[None, :]
    D = (x_e - p_e)**2 + wd[None, :]**2

    # ∂peak_k(x_i)/∂w_k = (1/π) · width_k / D_ik
    dpdw  = _PI_INV * wd[None, :] / D
    # ∂peak_k(x_i)/∂width_k = (1/π) · w_k · (D_ik - 2·width_k²) / D_ik²
    dpdwd = _PI_INV * w[None, :] * (D - 2.0 * wd[None, :]**2) / (D * D)

    m, k = dpdw.shape
    # Interleave: J[:, 0::2] = dpdw, J[:, 1::2] = dpdwd to match y's layout.
    return jnp.stack([dpdw, dpdwd], axis=2).reshape(m, 2 * k)


def H_analytic_full(y):
    """Full analytic Hessian for DMN15102LS.

    Exploits the residual's structure:
      ∂²r_i/∂w_j ∂w_k          = 0  (r is linear in each w_j)
      ∂²r_i/∂w_j ∂width_k      = δ_{jk} · (1/π) · ((x_i - pos_j)² - width_j²) / D_ij²
      ∂²r_i/∂width_j ∂width_k  = δ_{jk} · (1/π) · w_j · (-2 width_j) ·
                                 (3 D_ij - 4 width_j²) / D_ij³

    Correction matrix is therefore sparse with structure:
      - (w_j, w_k)         block: zero
      - (w_j, width_k)     block: diagonal in (j, k)
      - (width_j, width_k) block: diagonal in (j, k)
    Total nonzero entries: 2·n_peaks = 66 out of 66² = 4356 (1.5% density).
    """
    w  = y[0::2]; wd = y[1::2]
    x_e = _X_DATA[:, None]; p_e = _POSITIONS[None, :]
    diff = x_e - p_e
    D = diff**2 + wd[None, :]**2

    # Jacobian: one batched op (same as J_analytic above).
    dpdw  = _PI_INV * wd[None, :] / D
    dpdwd = _PI_INV * w[None, :] * (D - 2.0 * wd[None, :]**2) / (D * D)
    m, k = dpdw.shape
    J = jnp.stack([dpdw, dpdwd], axis=2).reshape(m, 2 * k)
    JTJ = J.T @ J

    # Sparse correction: only per-peak nonzero second-partials.
    r = jnp.sum(w[None, :] * _PI_INV * wd[None, :] / D, axis=1) - _Y_DATA
    d2_w_wd  = _PI_INV * (diff**2 - wd[None, :]**2) / (D * D)
    d2_wd_wd = _PI_INV * w[None, :] * (-2.0 * wd[None, :]) * \
               (3.0 * D - 4.0 * wd[None, :]**2) / (D**3)
    c_w_wd  = jnp.sum(r[:, None] * d2_w_wd,  axis=0)   # (k,)
    c_wd_wd = jnp.sum(r[:, None] * d2_wd_wd, axis=0)   # (k,)

    corr = jnp.zeros((2*k, 2*k), dtype=y.dtype)
    idx_w  = jnp.arange(0, 2*k, 2)
    idx_wd = jnp.arange(1, 2*k, 2)
    corr = corr.at[idx_w,  idx_wd].set(c_w_wd)
    corr = corr.at[idx_wd, idx_w].set(c_w_wd)
    corr = corr.at[idx_wd, idx_wd].set(c_wd_wd)

    # f(y) = sum(r²)  (no 1/2 factor in DMN15102LS), so H = 2(JᵀJ + corr).
    return 2.0 * (JTJ + corr)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    import statistics, time
    from sif2jax.cutest import DMN15102LS

    p = DMN15102LS()
    y0 = jnp.asarray(p.y0, jnp.float64)

    f = lambda y: jnp.sum(r_fn(y)**2)
    H_an  = jax.jit(H_analytic_full)
    H_jax = jax.jit(jax.hessian(f))

    H_a = H_an(y0); jax.block_until_ready(H_a)
    H_j = H_jax(y0); jax.block_until_ready(H_j)
    rel = float(jnp.max(jnp.abs(H_a - H_j))) / float(jnp.max(jnp.abs(H_j)))
    print(f"Analytic vs jax.hessian:  rel err = {rel:.2e}")

    def _time(fn, runs=30):
        for _ in range(3): jax.block_until_ready(fn(y0))
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); jax.block_until_ready(fn(y0))
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    t_a = _time(H_an); t_j = _time(H_jax)
    print(f"H_analytic_full :  {t_a*1e3:7.3f} ms")
    print(f"jax.hessian     :  {t_j*1e3:7.3f} ms")
    print(f"speedup         :  {t_j/t_a:5.1f}x")
