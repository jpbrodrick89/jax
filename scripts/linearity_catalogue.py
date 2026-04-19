#!/usr/bin/env python3
"""
Empirically catalogue JAX primitives and their linear arguments.

Strategy: for each (operation, arg_idx) pair construct a closure that fixes
all other arguments and apply jax.linear_transpose to it.  If that succeeds
the argument is linear.  No source-code inspection is used; all conclusions
come from JAX's own transpose machinery raising (or not) on the jaxpr.

Run:
    PYTHONPATH=/path/to/jax python scripts/linearity_catalogue.py
"""

import warnings
warnings.filterwarnings("ignore")

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax._src.lax import linalg as lax_linalg
from collections import defaultdict

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Primitive counts from AD registries
# ---------------------------------------------------------------------------
from jax._src.interpreters import ad
import jax._src.core as jax_core, gc

jvp_prims       = set(ad.primitive_jvps.keys())
transpose_prims = set(ad.primitive_transposes.keys())
fancy_prims     = set(getattr(ad, "fancy_transposes", {}).keys())
all_diff_prims  = jvp_prims | transpose_prims | fancy_prims
all_prims_mem   = {o for o in gc.get_objects() if isinstance(o, jax_core.Primitive)}

print("=== Primitive registry ===")
print(f"  Total Primitive objects live in memory : {len(all_prims_mem)}")
print(f"  Differentiable (jvp | transpose | fancy): {len(all_diff_prims)}")
print(f"    with JVP rules                        : {len(jvp_prims)}")
print(f"    with transpose rules                  : {len(transpose_prims)}")
print(f"    with fancy-transpose rules            : {len(fancy_prims)}")
print()

# ---------------------------------------------------------------------------
# Linearity test helper
# ---------------------------------------------------------------------------

def is_linear_in(fn, args, idx):
    """
    Return True iff fn(*args) is linear in args[idx], tested via
    jax.linear_transpose (raises on non-linear ops in the jaxpr).
    Only inexact (float/complex) arguments can be linear.
    """
    x0 = args[idx]
    if not (hasattr(x0, "dtype") and jnp.issubdtype(x0.dtype, jnp.inexact)):
        return False

    def f(x):
        a = list(args)
        a[idx] = x
        out = fn(*a)
        # sum all floating leaves to get a scalar (linear_transpose needs
        # a fixed-shape real/complex output)
        leaves = jax.tree.leaves(out)
        return sum(
            jnp.sum(l) for l in leaves
            if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.inexact)
        )

    try:
        out_ev = jax.eval_shape(f, x0)
        ct = jnp.ones(out_ev.shape, out_ev.dtype)
        jax.linear_transpose(f, x0)(ct)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test cases  (fn, args, name, category)
# Only inexact-dtype entries in args are tested for linearity.
# ---------------------------------------------------------------------------

_f  = jnp.array(1.5, jnp.float32)
_f2 = jnp.ones(3,    jnp.float32)
_f3 = jnp.ones((3,3),jnp.float32)
_cs = jnp.array(1+2j, jnp.complex64)
_cv = jnp.ones(3,     jnp.complex64)
_sym = jnp.eye(3, dtype=jnp.float32) * 2.0    # pos-def for cholesky/eigh
_tri = jnp.tril(jnp.ones((3,3),jnp.float32)) + jnp.eye(3,dtype=jnp.float32)

TEST_CASES = [
    # ---- unary elementwise ------------------------------------------------
    (lax.neg,   [_f],  "neg",   "elementwise"),
    (lax.abs,   [_f],  "abs",   "elementwise"),
    (lax.sign,  [_f],  "sign",  "elementwise"),
    (lax.floor, [_f],  "floor", "elementwise"),
    (lax.ceil,  [_f],  "ceil",  "elementwise"),
    (lax.round, [_f],  "round", "elementwise"),
    (lax.exp,   [_f],  "exp",   "elementwise"),
    (lax.exp2,  [_f],  "exp2",  "elementwise"),
    (lax.log,   [_f],  "log",   "elementwise"),
    (lax.log1p, [_f],  "log1p", "elementwise"),
    (lax.expm1, [_f],  "expm1", "elementwise"),
    (lax.sqrt,  [_f],  "sqrt",  "elementwise"),
    (lax.cbrt,  [_f],  "cbrt",  "elementwise"),
    (lax.rsqrt, [_f],  "rsqrt", "elementwise"),
    (lax.sin,   [_f],  "sin",   "elementwise"),
    (lax.cos,   [_f],  "cos",   "elementwise"),
    (lax.tan,   [_f],  "tan",   "elementwise"),
    (lax.asin,  [_f],  "asin",  "elementwise"),
    (lax.acos,  [_f],  "acos",  "elementwise"),
    (lax.atan,  [_f],  "atan",  "elementwise"),
    (lax.sinh,  [_f],  "sinh",  "elementwise"),
    (lax.cosh,  [_f],  "cosh",  "elementwise"),
    (lax.tanh,  [_f],  "tanh",  "elementwise"),
    (lax.asinh, [_f],  "asinh", "elementwise"),
    (lax.acosh, [jnp.array(2.0,jnp.float32)], "acosh", "elementwise"),
    (lax.atanh, [jnp.array(0.5,jnp.float32)], "atanh", "elementwise"),
    (lax.erf,   [_f],  "erf",       "special"),
    (lax.erfc,  [_f],  "erfc",      "special"),
    (lax.erf_inv,[jnp.array(0.5,jnp.float32)], "erf_inv", "special"),
    (lambda x: lax.bessel_i0e(x), [_f], "bessel_i0e", "special"),
    (lambda x: lax.bessel_i1e(x), [_f], "bessel_i1e", "special"),
    (lambda x: lax.polygamma(0,x), [_f], "polygamma",  "special"),
    (lambda x: lax.zeta(2.0, x),   [_f], "zeta",       "special"),
    (lambda x: lax.integer_pow(x, 2), [_f], "integer_pow(n=2)", "elementwise"),
    (lambda x: lax.integer_pow(x, 3), [_f], "integer_pow(n=3)", "elementwise"),
    (lax.is_finite, [_f], "is_finite", "elementwise"),
    (lambda x: lax.reduce_precision(x, 8, 23), [_f], "reduce_precision", "elementwise"),
    (lambda x: lax.convert_element_type(x, jnp.float64), [_f],
                                               "convert_element_type", "misc"),

    # ---- binary elementwise -----------------------------------------------
    (lax.add,  [_f,_f], "add",  "elementwise"),
    (lax.sub,  [_f,_f], "sub",  "elementwise"),
    (lax.mul,  [_f,_f], "mul",  "elementwise"),
    (lax.div,  [_f,_f], "div",  "elementwise"),
    (lax.rem,  [_f,_f], "rem",  "elementwise"),
    (lax.max,  [_f,_f], "max",  "elementwise"),
    (lax.min,  [_f,_f], "min",  "elementwise"),
    (lax.pow,  [_f,_f], "pow",  "elementwise"),
    (lax.atan2,[_f,_f], "atan2","elementwise"),
    (lax.igamma, [_f,_f], "igamma",  "special"),
    (lax.igammac,[_f,_f], "igammac", "special"),
    (lax.betainc,[_f,_f,_f], "betainc","special"),
    (lax.select,[jnp.array(True),_f,_f], "select", "misc"),
    (lax.complex,[_f,_f], "complex", "complex"),

    # ---- complex -------------------------------------------------------------
    (lax.real, [_cs], "real",        "complex"),
    (lax.imag, [_cs], "imag",        "complex"),
    (jnp.conj, [_cs], "conj",        "complex"),
    (lax.abs,  [_cs], "abs(complex)","complex"),

    # ---- reductions ----------------------------------------------------------
    (lambda x: lax.reduce_sum(x,[0]),           [_f2], "reduce_sum",        "reduction"),
    (lambda x: lax.reduce_prod(x,[0]),           [_f2], "reduce_prod",       "reduction"),
    (lambda x: lax.reduce_max(x,[0]),            [_f2], "reduce_max",        "reduction"),
    (lambda x: lax.reduce_min(x,[0]),            [_f2], "reduce_min",        "reduction"),
    (lambda x: lax.cumsum(x, axis=0),            [_f2], "cumsum",            "reduction"),
    (lambda x: lax.cumprod(x, axis=0),           [_f2], "cumprod",           "reduction"),
    (lambda x: lax.associative_scan(lax.add,x),  [_f2], "assoc_scan(add)",   "reduction"),
    (lambda x: lax.associative_scan(lax.mul,x),  [_f2], "assoc_scan(mul)",   "reduction"),

    # ---- shape / indexing ----------------------------------------------------
    (lambda x: lax.reshape(x,(1,3)),             [_f2], "reshape",           "shape"),
    (lambda x: lax.transpose(x,(1,0)),           [_f3], "transpose",         "shape"),
    (lambda x: lax.rev(x,[0]),                   [_f2], "rev",               "shape"),
    (lambda x: lax.squeeze(x,[0]),               [_f2.reshape(1,3)], "squeeze","shape"),
    (lambda x: lax.broadcast_in_dim(x,(2,3),(1,)),[_f2], "broadcast_in_dim", "shape"),
    (lambda x,y: lax.concatenate([x,y],0),       [_f2,_f2], "concatenate",   "shape"),
    (lambda x: lax.slice(x,[0],[2]),              [_f2], "slice",             "indexing"),
    (lambda x: lax.dynamic_slice(x,[jnp.array(0,jnp.int32)],[2]), [_f2],
                                                         "dynamic_slice",     "indexing"),
    (lambda x: lax.pad(x,jnp.array(0.,jnp.float32),[(1,1,0)]), [_f2],
                                                         "pad",               "shape"),
    (lambda x,y: lax.dynamic_update_slice(x,y,[jnp.array(0,jnp.int32)]),
     [_f2,jnp.ones(2,jnp.float32)],                     "dynamic_update_slice","indexing"),

    # ---- gather / scatter ----------------------------------------------------
    (lambda x: lax.gather(
        x, jnp.array([[0],[2]],jnp.int32),
        lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
        slice_sizes=(1,)),
     [_f2],                                              "gather",            "indexing"),
    (lambda x,y: lax.scatter_add(
        x, jnp.array([[0],[2]],jnp.int32), y,
        lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
     [_f2,jnp.ones(2,jnp.float32)],                     "scatter_add",       "indexing"),

    # ---- dot / matmul --------------------------------------------------------
    (lax.dot,  [_f2,_f2], "dot",  "linalg"),
    (lambda a,b: lax.dot_general(a,b,(([0],[0]),([],[]))), [_f2,_f2],
                                                  "dot_general",             "linalg"),
    (lambda a,b: jnp.einsum("ij,jk->ik",a,b), [_f3,_f3],
                                                  "einsum(matmul)",          "linalg"),

    # ---- convolution ---------------------------------------------------------
    (lambda x,k: lax.conv_general_dilated(
        x.reshape(1,1,3), k.reshape(1,1,3), (1,), "SAME"),
     [_f2, jnp.ones(3,jnp.float32)],             "conv(input+kernel)",      "linalg"),

    # ---- linear algebra (factorizations / decompositions) --------------------
    (lambda x: lax_linalg.cholesky(x),   [_sym], "cholesky",       "linalg"),
    (lambda x: jnpla.inv(x),             [_sym], "inv",            "linalg"),

    # triangular_solve: linear in rhs (b), not in matrix (a)
    (lambda b: lax_linalg.triangular_solve(
        _tri, b, lower=True, transpose_a=False,
        unit_diagonal=False, left_side=True),
     [_f2.reshape(3,1)],                          "triangular_solve(rhs)",  "linalg"),
    (lambda a: lax_linalg.triangular_solve(
        a, _f2.reshape(3,1), lower=True, transpose_a=False,
        unit_diagonal=False, left_side=True),
     [_tri],                                      "triangular_solve(mat)",  "linalg"),

    # tridiagonal_solve: linear in rhs
    (lambda r: lax_linalg.tridiagonal_solve(
        jnp.ones(3,jnp.float32)*0.5,
        jnp.ones(3,jnp.float32)*3.0,
        jnp.ones(3,jnp.float32)*0.5, r),
     [jnp.ones((3,1),jnp.float32)],              "tridiagonal_solve(rhs)", "linalg"),

    # lu_solve: linear in rhs
    (lambda b: lax_linalg.lu_solve(
        *lax_linalg.lu(jnp.eye(3,dtype=jnp.float32)*2)[:2], b, trans=0),
     [jnp.ones(3,jnp.float32)],                  "lu_solve(rhs)",          "linalg"),

    # linalg.solve: linear in rhs, not in matrix
    (lambda b: jnpla.solve(_sym, b),
     [jnp.ones(3,jnp.float32)],                  "linalg.solve(rhs)",      "linalg"),
    (lambda a: jnpla.solve(a, jnp.ones(3,jnp.float32)),
     [_sym],                                      "linalg.solve(mat)",      "linalg"),

    (lambda x: lax_linalg.lu(x),                 [_f3], "lu",              "linalg"),
    (lambda x: lax_linalg.qr(x,full_matrices=False),[_f3],"qr",            "linalg"),
    (lambda x: lax_linalg.eigh(x+x.T),           [_f3], "eigh",            "linalg"),
    (lambda x: lax_linalg.eig(x),                [_f3], "eig",             "linalg"),
    (lambda x: lax_linalg.svd(x,full_matrices=False),[_f3],"svd",          "linalg"),
    (lambda x: lax_linalg.schur(x),              [_f3], "schur",           "linalg"),

    # ---- FFT -----------------------------------------------------------------
    (lambda x: jnp.fft.fft(x),                  [_cv],  "fft",             "spectral"),
    (lambda x: jnp.fft.ifft(x),                 [_cv],  "ifft",            "spectral"),
    (lambda x: jnp.fft.rfft(x),                 [_f2],  "rfft",            "spectral"),
    (lambda x: jnp.fft.irfft(jnp.fft.rfft(x),n=len(x)),[_f2],"irfft",    "spectral"),
    (lambda x: jnp.fft.fftn(x.reshape(1,3)),    [_f2],  "fftn",            "spectral"),

    # ---- control flow --------------------------------------------------------
    (lambda x: lax.cond(jnp.array(True),
                        lambda: x*2., lambda: x*3.),            [_f],
                                                  "cond",               "control_flow"),
    (lambda x: lax.while_loop(
        lambda s: s[0]<3., lambda s: (s[0]+1., s[1]),
        (jnp.array(0.,jnp.float32), x))[1],      [_f],
                                                  "while_loop",         "control_flow"),
    (lambda x: lax.scan(
        lambda c,a: (c, a+c),
        jnp.array(0.,jnp.float32), x)[1],        [_f2],
                                                  "scan",               "control_flow"),

    # ---- misc ----------------------------------------------------------------
    (lambda x: lax.sort(x),              [_f2], "sort",             "misc"),
    (lambda x: lax.top_k(x,2)[0],       [_f2], "top_k",            "misc"),
    (lambda x: lax.stop_gradient(x),    [_f],  "stop_gradient",    "misc"),
    (lambda x: jnp.where(jnp.array(True), x, jnp.array(0.,jnp.float32)),
                                         [_f],  "where",            "misc"),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = defaultdict(list)
errors  = []

print("=== Empirical linearity tests ===")
print(f"  (testing {len(TEST_CASES)} named operations)\n")
print(f"{'name':<32} {'nfloat':>6}  {'linear args'}")
print("-" * 58)

for fn, args, name, category in TEST_CASES:
    args = list(args)
    float_indices = [
        i for i,a in enumerate(args)
        if hasattr(a,"dtype") and jnp.issubdtype(a.dtype, jnp.inexact)
    ]
    linear_args = []
    for i in float_indices:
        try:
            if is_linear_in(fn, args, i):
                linear_args.append(i)
        except Exception as e:
            errors.append((name, i, str(e)))

    tag = str(linear_args) if linear_args else "—"
    print(f"  {name:<30} {len(float_indices):>6}  {tag}")
    results[category].append((name, linear_args))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

all_ops      = sum(len(v) for v in results.values())
ops_with_lin = sum(1 for v in results.values() for _,la in v if la)

print(f"""
{'='*58}
SUMMARY
{'='*58}
Primitive registry:
  Total Primitive objects in memory         : {len(all_prims_mem)}
  Differentiable (jvp | transpose | fancy)  : {len(all_diff_prims)}
    - with JVP rules                        : {len(jvp_prims)}
    - with transpose rules                  : {len(transpose_prims)}
    - with fancy-transpose rules            : {len(fancy_prims)}

Empirical linearity ({all_ops} operations tested):
  With ≥1 linear argument : {ops_with_lin}
  With no linear argument  : {all_ops - ops_with_lin}
""")

print("=== Linear operations by category ===\n")
for cat in sorted(results):
    ops     = results[cat]
    lin_ops = [(n,la) for n,la in ops if la]
    print(f"{cat.upper()}  ({len(lin_ops)}/{len(ops)} ops have linear args)")
    for name, la in lin_ops:
        labels = ", ".join(f"arg{i}" for i in la)
        print(f"    {name:<32}  linear in: {labels}")
    print()

if errors:
    print(f"=== Errors during testing ({len(errors)}) ===")
    for name, idx, msg in errors[:10]:
        print(f"  {name}[arg{idx}]: {msg[:100]}")
