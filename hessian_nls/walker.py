"""Minimal edge-pushing walker for the residual-correction branch.

Computes  Σ_i r̄_i · ∇²r_i  for a residual function r(x), given the
cotangent vector r̄.  Walks the residual sub-jaxpr in two sweeps:

    forward   — propagates (primal, J = ∂var/∂x) per intermediate.
    backward  — at each primitive, accumulates the symmetric Hessian
                contribution (rank-1 for elementwise nonlinear, rank-2
                for bilinear) weighted by that primitive's cotangent,
                and propagates cotangents to its inputs.

This is the canonical edge-pushing scheme.  In this prototype Jacobians
are *dense*; the per-primitive accumulator is `einsum('i,ij,ik->jk',
cot, J, J)`, which is O(m · n²) per primitive.  Total walker cost on a
dense problem is therefore O(P · m · n²), strictly larger than
`jax.hessian(φ)`'s O(n · P) for n ~ m.

The walker is therefore *not* competitive with stock JAX for dense
problems at scale — it ties or wins at small n (where dense matmul
overhead matters less than per-pass orchestration) and loses at larger
n.  This matches expectations: edge-pushing's structural advantage
comes from *sparsity-aware* Jacobian storage, where `J[i,:]` has only
`nnz_i ≪ n` nonzeros and the rank-1/rank-2 contributions cost
`O(nnz_i²)` instead of `O(n²)`.

Users with a sparse-Jacobian propagator (BCOO/CSR/ELLPACK) can replace
the dense ndarray operations in `forward_rules` and `backward_rules`
with sparse equivalents while keeping the driver in `edge_push_correction`
unchanged.  The driver makes one `forward_rules[prim_name](...)` and
one `backward_rules[prim_name](...)` call per primitive, with no
implicit assumption that Jacobians are dense.

Currently supported primitives (sufficient for the test battery):
    linear:        add sub neg convert_element_type reduce_sum
                   broadcast_in_dim reshape squeeze transpose
                   slice pad concatenate iota
    bilinear:      mul dot_general
    elemwise NL:   integer_pow sin cos tanh exp log sqrt

Anything outside this set raises NotImplementedError; callers fall back
to the stock jax.hessian-based correction.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Literal, Var


# ---- helpers ---------------------------------------------------------

def _val(env, v):
    """Resolve a Var or Literal to its concrete value."""
    return v.val if isinstance(v, Literal) else env[v][0]


def _jac(env, v):
    """Resolve Jacobian wrt the original input (zero for literals)."""
    if isinstance(v, Literal):
        # Literal has no x-dependence; broadcast a zero Jacobian.
        return None
    return env[v][1]


def _zero_like_jac(shape, n, dtype):
    return jnp.zeros(shape + (n,), dtype=dtype)


def _is_x_dep(j):
    return j is not None


def _broadcast_jac(j, target_shape):
    """Broadcast a Jacobian's value-axes to a target shape, leaving the
    last (input) axis intact."""
    if j is None:
        return None
    return jnp.broadcast_to(j, target_shape + (j.shape[-1],))


# ---- forward rules ---------------------------------------------------
# Each rule receives (eqn, env_lookup_value, env_lookup_jac, n) and returns
# (primal_out, J_out) with J_out shape = primal_out.shape + (n,).

def _fwd_add(primals, jacs, params):
    a, b = primals
    out = a + b
    if jacs[0] is None and jacs[1] is None:
        return out, None
    ja = jacs[0] if jacs[0] is not None else _zero_like_jac(jnp.shape(a), jacs[1].shape[-1], jacs[1].dtype)
    jb = jacs[1] if jacs[1] is not None else _zero_like_jac(jnp.shape(b), jacs[0].shape[-1], jacs[0].dtype)
    # broadcast for shape mismatch
    out_shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    ja = _broadcast_jac(ja, out_shape)
    jb = _broadcast_jac(jb, out_shape)
    return out, ja + jb


def _fwd_sub(primals, jacs, params):
    a, b = primals
    out = a - b
    if jacs[0] is None and jacs[1] is None:
        return out, None
    n = (jacs[0] if jacs[0] is not None else jacs[1]).shape[-1]
    dt = (jacs[0] if jacs[0] is not None else jacs[1]).dtype
    ja = jacs[0] if jacs[0] is not None else _zero_like_jac(jnp.shape(a), n, dt)
    jb = jacs[1] if jacs[1] is not None else _zero_like_jac(jnp.shape(b), n, dt)
    out_shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    ja = _broadcast_jac(ja, out_shape)
    jb = _broadcast_jac(jb, out_shape)
    return out, ja - jb


def _fwd_neg(primals, jacs, params):
    return -primals[0], (None if jacs[0] is None else -jacs[0])


def _fwd_convert(primals, jacs, params):
    a = primals[0].astype(params['new_dtype'])
    j = None if jacs[0] is None else jacs[0].astype(params['new_dtype'])
    return a, j


def _fwd_mul(primals, jacs, params):
    a, b = primals
    out = a * b
    if jacs[0] is None and jacs[1] is None:
        return out, None
    out_shape = jnp.shape(out)
    n = (jacs[0] if jacs[0] is not None else jacs[1]).shape[-1]
    dt = out.dtype
    ja = jacs[0] if jacs[0] is not None else _zero_like_jac(jnp.shape(a), n, dt)
    jb = jacs[1] if jacs[1] is not None else _zero_like_jac(jnp.shape(b), n, dt)
    ja = _broadcast_jac(ja, out_shape)
    jb = _broadcast_jac(jb, out_shape)
    a_b = jnp.broadcast_to(a, out_shape)[..., None]
    b_b = jnp.broadcast_to(b, out_shape)[..., None]
    return out, a_b * jb + b_b * ja


def _fwd_integer_pow(primals, jacs, params):
    y = params['y']
    a = primals[0]
    out = a ** y
    if jacs[0] is None:
        return out, None
    deriv = y * (a ** (y - 1)) if y != 0 else jnp.zeros_like(a)
    return out, deriv[..., None] * jacs[0]


def _fwd_unary_elem(g, gprime):
    def rule(primals, jacs, params):
        a = primals[0]
        out = g(a)
        if jacs[0] is None:
            return out, None
        return out, gprime(a)[..., None] * jacs[0]
    return rule


_fwd_sin   = _fwd_unary_elem(jnp.sin, jnp.cos)
_fwd_cos   = _fwd_unary_elem(jnp.cos, lambda a: -jnp.sin(a))
_fwd_tanh  = _fwd_unary_elem(jnp.tanh, lambda a: 1 - jnp.tanh(a)**2)
_fwd_exp   = _fwd_unary_elem(jnp.exp, jnp.exp)
_fwd_log   = _fwd_unary_elem(jnp.log, lambda a: 1.0 / a)
_fwd_sqrt  = _fwd_unary_elem(jnp.sqrt, lambda a: 0.5 / jnp.sqrt(a))


def _fwd_dot_general(primals, jacs, params):
    a, b = primals
    dn = params['dimension_numbers']
    out = jax.lax.dot_general(a, b, dn,
                              precision=params.get('precision'),
                              preferred_element_type=params.get('preferred_element_type'))
    if jacs[0] is None and jacs[1] is None:
        return out, None
    n = (jacs[0] if jacs[0] is not None else jacs[1]).shape[-1]
    # Use jax.jvp under the hood for correctness & dtype handling.
    a_dot = jacs[0] if jacs[0] is not None else jnp.zeros(jnp.shape(a) + (n,), out.dtype)
    b_dot = jacs[1] if jacs[1] is not None else jnp.zeros(jnp.shape(b) + (n,), out.dtype)
    # contract over input axis (the LAST axis of jac is input dim n;
    # we want to map over it).  vmap along last axis of each:
    def jvp_one(da, db):
        _, jvp_out = jax.jvp(
            lambda x, y: jax.lax.dot_general(x, y, dn,
                                              precision=params.get('precision'),
                                              preferred_element_type=params.get('preferred_element_type')),
            (a, b), (da, db))
        return jvp_out
    j_out = jax.vmap(jvp_one, in_axes=(-1, -1), out_axes=-1)(a_dot, b_dot)
    return out, j_out


def _fwd_reduce_sum(primals, jacs, params):
    a = primals[0]
    axes = params['axes']
    out = jnp.sum(a, axis=axes)
    if jacs[0] is None:
        return out, None
    j = jnp.sum(jacs[0], axis=axes)
    return out, j


def _fwd_broadcast(primals, jacs, params):
    a = primals[0]
    shape = params['shape']
    bdims = params['broadcast_dimensions']
    out = jax.lax.broadcast_in_dim(a, shape, bdims)
    if jacs[0] is None:
        return out, None
    j = jax.lax.broadcast_in_dim(jacs[0], shape + (jacs[0].shape[-1],),
                                 tuple(bdims) + (len(shape),))
    return out, j


def _fwd_reshape(primals, jacs, params):
    a = primals[0]
    new_sizes = params['new_sizes']
    dimensions = params.get('dimensions')
    out = jax.lax.reshape(a, new_sizes, dimensions)
    if jacs[0] is None:
        return out, None
    n = jacs[0].shape[-1]
    if dimensions is not None:
        j = jax.lax.transpose(jacs[0], tuple(dimensions) + (jacs[0].ndim - 1,))
    else:
        j = jacs[0]
    j = j.reshape(tuple(new_sizes) + (n,))
    return out, j


def _fwd_squeeze(primals, jacs, params):
    a = primals[0]
    dims = params['dimensions']
    out = jnp.squeeze(a, axis=dims)
    if jacs[0] is None:
        return out, None
    return out, jnp.squeeze(jacs[0], axis=dims)


def _fwd_transpose(primals, jacs, params):
    a = primals[0]
    perm = params['permutation']
    out = jnp.transpose(a, perm)
    if jacs[0] is None:
        return out, None
    return out, jnp.transpose(jacs[0], tuple(perm) + (jacs[0].ndim - 1,))


def _fwd_slice(primals, jacs, params):
    a = primals[0]
    start = params['start_indices']
    limit = params['limit_indices']
    strides = params.get('strides')
    out = jax.lax.slice(a, start, limit, strides)
    if jacs[0] is None:
        return out, None
    j_start  = tuple(start) + (0,)
    j_limit  = tuple(limit) + (jacs[0].shape[-1],)
    j_stride = None if strides is None else tuple(strides) + (1,)
    j = jax.lax.slice(jacs[0], j_start, j_limit, j_stride)
    return out, j


def _fwd_pad(primals, jacs, params):
    a, pv = primals
    cfg = params['padding_config']
    out = jax.lax.pad(a, pv, cfg)
    if jacs[0] is None:
        return out, None
    # Pad the Jacobian's value-axes with zeros (no padding on the input axis).
    jcfg = list(cfg) + [(0, 0, 0)]
    j = jax.lax.pad(jacs[0], jnp.array(0, jacs[0].dtype), jcfg)
    return out, j


def _fwd_concat(primals, jacs, params):
    dim = params['dimension']
    out = jnp.concatenate(primals, axis=dim)
    if all(j is None for j in jacs):
        return out, None
    n = next(j for j in jacs if j is not None).shape[-1]
    dt = out.dtype
    js = [j if j is not None else jnp.zeros(jnp.shape(p) + (n,), dt)
          for j, p in zip(jacs, primals)]
    return out, jnp.concatenate(js, axis=dim)


def _fwd_iota(primals, jacs, params):
    out = jax.lax.iota(params['dtype'], params['shape'][params['dimension']])
    if 'shape' in params and len(params['shape']) > 1:
        out = jax.lax.broadcasted_iota(params['dtype'], params['shape'],
                                       params['dimension'])
    return out, None


FORWARD_RULES = {
    'add': _fwd_add,
    'sub': _fwd_sub,
    'neg': _fwd_neg,
    'convert_element_type': _fwd_convert,
    'mul': _fwd_mul,
    'integer_pow': _fwd_integer_pow,
    'sin': _fwd_sin,
    'cos': _fwd_cos,
    'tanh': _fwd_tanh,
    'exp': _fwd_exp,
    'log': _fwd_log,
    'sqrt': _fwd_sqrt,
    'dot_general': _fwd_dot_general,
    'reduce_sum': _fwd_reduce_sum,
    'broadcast_in_dim': _fwd_broadcast,
    'reshape': _fwd_reshape,
    'squeeze': _fwd_squeeze,
    'transpose': _fwd_transpose,
    'slice': _fwd_slice,
    'pad': _fwd_pad,
    'concatenate': _fwd_concat,
    'iota': _fwd_iota,
}


# ---- backward rules --------------------------------------------------
# Each rule:
#   - reads input primals + Jacobians from env,
#   - reads cotangent_out from cotangents[output_var],
#   - accumulates Hessian contribution into accumulator,
#   - returns a list of cotangent contributions for each input (None for
#     non-x-dependent inputs / Literals).
# H accumulator is an n×n ndarray; we replace it functionally.

def _einsum_rank1(c, J, accumulator):
    """Accumulate  sum_i c[i] · J[i,:] ⊗ J[i,:]   into accumulator (n,n)."""
    # c and J share leading axes; flatten them.
    c_flat = c.reshape(-1)
    J_flat = J.reshape(-1, J.shape[-1])
    return accumulator + jnp.einsum('i,ij,ik->jk', c_flat, J_flat, J_flat)


def _einsum_rank2_sym(c, Ja, Jb, accumulator):
    """Accumulate  sum_i c[i] · (Ja[i] ⊗ Jb[i] + Jb[i] ⊗ Ja[i])."""
    c_flat = c.reshape(-1)
    Ja_flat = Ja.reshape(-1, Ja.shape[-1])
    Jb_flat = Jb.reshape(-1, Jb.shape[-1])
    cross = jnp.einsum('i,ij,ik->jk', c_flat, Ja_flat, Jb_flat)
    return accumulator + cross + cross.T


def _bw_linear(eqn, env, cot_out, accum):
    """No Hessian contribution; cotangent passes through (with sign/etc.)."""
    name = eqn.primitive.name
    if name == 'add':
        return [cot_out, cot_out], accum
    if name == 'sub':
        return [cot_out, -cot_out], accum
    if name == 'neg':
        return [-cot_out], accum
    if name == 'convert_element_type':
        return [cot_out.astype(eqn.invars[0].aval.dtype)], accum
    raise AssertionError(name)


def _bw_mul(eqn, env, cot_out, accum):
    a, b = eqn.invars
    ja, jb = _jac(env, a), _jac(env, b)
    aval, bval = _val(env, a), _val(env, b)
    if _is_x_dep(ja) and _is_x_dep(jb):
        accum = _einsum_rank2_sym(cot_out, ja, jb, accum)
    cot_a = cot_out * jnp.broadcast_to(bval, jnp.shape(cot_out))
    cot_b = cot_out * jnp.broadcast_to(aval, jnp.shape(cot_out))
    cots = []
    for v, ct in zip(eqn.invars, [cot_a, cot_b]):
        cots.append(None if isinstance(v, Literal) else ct)
    return cots, accum


def _bw_integer_pow(eqn, env, cot_out, accum):
    y = eqn.params['y']
    (a,) = eqn.invars
    ja = _jac(env, a)
    aval = _val(env, a)
    if _is_x_dep(ja) and y >= 2:
        # second derivative = y(y-1) a^(y-2)
        gpp = y * (y - 1) * (aval ** (y - 2))
        accum = _einsum_rank1(cot_out * gpp, ja, accum)
    deriv = y * (aval ** (y - 1)) if y != 0 else jnp.zeros_like(aval)
    return [cot_out * deriv], accum


def _bw_unary_elem(g, gprime, gprimeprime):
    def rule(eqn, env, cot_out, accum):
        (a,) = eqn.invars
        ja = _jac(env, a)
        aval = _val(env, a)
        if _is_x_dep(ja):
            accum = _einsum_rank1(cot_out * gprimeprime(aval), ja, accum)
        return [cot_out * gprime(aval)], accum
    return rule


_bw_sin  = _bw_unary_elem(jnp.sin,  jnp.cos,                 lambda a: -jnp.sin(a))
_bw_cos  = _bw_unary_elem(jnp.cos,  lambda a: -jnp.sin(a),    lambda a: -jnp.cos(a))
_bw_tanh = _bw_unary_elem(jnp.tanh, lambda a: 1-jnp.tanh(a)**2,
                          lambda a: -2*jnp.tanh(a)*(1-jnp.tanh(a)**2))
_bw_exp  = _bw_unary_elem(jnp.exp,  jnp.exp,                  jnp.exp)
_bw_log  = _bw_unary_elem(jnp.log,  lambda a: 1.0/a,          lambda a: -1.0/(a*a))
_bw_sqrt = _bw_unary_elem(jnp.sqrt, lambda a: 0.5/jnp.sqrt(a),
                          lambda a: -0.25/(a**1.5))


def _bw_dot_general(eqn, env, cot_out, accum):
    a, b = eqn.invars
    aval, bval = _val(env, a), _val(env, b)
    ja, jb = _jac(env, a), _jac(env, b)
    dn = eqn.params['dimension_numbers']

    # Accumulate rank-2 cross term if both x-dependent.
    if _is_x_dep(ja) and _is_x_dep(jb):
        # General handling: contract J_a (shape sa+(n,)) and J_b along the
        # original contracting axes, but produce an outer-product over n.
        # Use vmap to unroll the n axis.
        n = ja.shape[-1]
        def cross_one(ja_col, jb_col):
            # rank-2 sym piece per input direction:
            # c · (ja_col⊗jb_col + jb_col⊗ja_col)  contracted into output via dn.
            # The output contribution to (i, j) of H is:
            # sum_o cot_out[o] · (∂out_o/∂a-direction-i)·(∂out_o/∂b-direction-j)+sym.
            d_a = jax.lax.dot_general(ja_col, bval, dn)
            d_b = jax.lax.dot_general(aval, jb_col, dn)
            return jnp.sum(cot_out * d_a), jnp.sum(cot_out * d_b)
        # We need outer product over (i, j) where i indexes the n-axis of ja
        # and j indexes the n-axis of jb. Compute the matrix
        #    M[i,j] = sum_o cot_out[o] · D_a_to_o[i] · D_b_to_o[j]  (and sym).
        # Where D_a_to_o[i] = ∂out_o / ∂(input direction i for arg a).
        # Equivalent to: contract cot_out with bval (giving "vjp"-like vector
        # over a's shape), then dot with ja_col elementwise summed.
        # Cleanest: vmap jvp along the n axis to materialise full output Jacobians.
        def out_jvp(a_dot, b_dot):
            _, jvp = jax.jvp(
                lambda x, y: jax.lax.dot_general(x, y, dn),
                (aval, bval), (a_dot, b_dot))
            return jvp
        # ja[..., i] is direction i of input a; want J_out_a [output_shape, n].
        zeros_jb = jnp.zeros_like(jb)
        zeros_ja = jnp.zeros_like(ja)
        J_out_from_a = jax.vmap(out_jvp, in_axes=(-1, -1), out_axes=-1)(ja, zeros_jb)
        J_out_from_b = jax.vmap(out_jvp, in_axes=(-1, -1), out_axes=-1)(zeros_ja, jb)
        accum = _einsum_rank2_sym(cot_out, J_out_from_a, J_out_from_b, accum)
        # Wait: that's ja⊗jb + jb⊗ja using J_out_from_a and J_out_from_b which
        # are the OUTPUT Jacobians along the a- and b-tangent directions — same
        # rank-2 structure, same formula.

    # Cotangent VJP through dot_general (use jax.vjp for correctness).
    _, vjp_fn = jax.vjp(
        lambda x, y: jax.lax.dot_general(x, y, dn,
                                          precision=eqn.params.get('precision'),
                                          preferred_element_type=eqn.params.get('preferred_element_type')),
        aval, bval)
    cot_a, cot_b = vjp_fn(cot_out)
    out_cots = []
    for v, ct in zip(eqn.invars, [cot_a, cot_b]):
        out_cots.append(None if isinstance(v, Literal) else ct)
    return out_cots, accum


def _bw_reduce_sum(eqn, env, cot_out, accum):
    (a,) = eqn.invars
    aval = _val(env, a)
    axes = eqn.params['axes']
    cot = jnp.broadcast_to(jnp.expand_dims(cot_out, axes), jnp.shape(aval))
    return [cot], accum


def _bw_broadcast(eqn, env, cot_out, accum):
    (a,) = eqn.invars
    bdims = eqn.params['broadcast_dimensions']
    shape = eqn.params['shape']
    a_shape = jnp.shape(_val(env, a))
    # Sum out broadcast axes (those NOT in bdims) and squeeze size-1 axes.
    sum_axes = tuple(i for i in range(len(shape)) if i not in bdims)
    cot = jnp.sum(cot_out, axis=sum_axes) if sum_axes else cot_out
    cot = cot.reshape(a_shape)
    return [cot], accum


def _bw_reshape(eqn, env, cot_out, accum):
    (a,) = eqn.invars
    aval = _val(env, a)
    dims = eqn.params.get('dimensions')
    if dims is not None:
        # The forward did transpose+reshape; reverse it.
        cot = cot_out.reshape(jnp.shape(jnp.transpose(aval, dims)))
        inv_perm = tuple(np.argsort(np.array(dims)))
        cot = jnp.transpose(cot, inv_perm)
    else:
        cot = cot_out.reshape(jnp.shape(aval))
    return [cot], accum


def _bw_squeeze(eqn, env, cot_out, accum):
    (a,) = eqn.invars
    aval = _val(env, a)
    dims = eqn.params['dimensions']
    cot = jnp.expand_dims(cot_out, dims).reshape(jnp.shape(aval))
    return [cot], accum


def _bw_transpose(eqn, env, cot_out, accum):
    (a,) = eqn.invars
    perm = eqn.params['permutation']
    inv = tuple(np.argsort(np.array(perm)))
    return [jnp.transpose(cot_out, inv)], accum


def _bw_slice(eqn, env, cot_out, accum):
    (a,) = eqn.invars
    aval = _val(env, a)
    start = eqn.params['start_indices']
    limit = eqn.params['limit_indices']
    strides = eqn.params.get('strides')
    if strides is not None and any(s != 1 for s in strides):
        raise NotImplementedError("strided slice not supported")
    # cotangent is zero outside the slice; pad to full shape.
    pad_cfg = []
    for i, (s, l) in enumerate(zip(start, limit)):
        before = int(s)
        after = int(aval.shape[i] - l)
        pad_cfg.append((before, after, 0))
    cot = jax.lax.pad(cot_out, jnp.array(0, cot_out.dtype), pad_cfg)
    return [cot], accum


def _bw_pad(eqn, env, cot_out, accum):
    a, _pv = eqn.invars
    cfg = eqn.params['padding_config']
    aval = _val(env, a)
    # Slice out the unpadded region (assumes no interior padding, no negative pads).
    start  = tuple(int(c[0]) for c in cfg)
    limit  = tuple(s - int(c[1]) for s, c in zip(jnp.shape(cot_out), cfg))
    cot = jax.lax.slice(cot_out, start, limit)
    return [cot, None], accum


def _bw_concat(eqn, env, cot_out, accum):
    dim = eqn.params['dimension']
    sizes = [jnp.shape(_val(env, v))[dim] for v in eqn.invars]
    splits = np.cumsum(sizes[:-1]).tolist()
    parts = jnp.split(cot_out, splits, axis=dim)
    cots = []
    for v, p in zip(eqn.invars, parts):
        cots.append(None if isinstance(v, Literal) else p)
    return cots, accum


def _bw_neg(eqn, env, cot_out, accum):
    return [-cot_out], accum


def _bw_convert(eqn, env, cot_out, accum):
    return [cot_out.astype(eqn.invars[0].aval.dtype)], accum


def _bw_iota(eqn, env, cot_out, accum):
    return [], accum


BACKWARD_RULES = {
    'add': lambda eqn, env, c, h: _bw_linear(eqn, env, c, h),
    'sub': lambda eqn, env, c, h: _bw_linear(eqn, env, c, h),
    'neg': _bw_neg,
    'convert_element_type': _bw_convert,
    'mul': _bw_mul,
    'integer_pow': _bw_integer_pow,
    'sin': _bw_sin,
    'cos': _bw_cos,
    'tanh': _bw_tanh,
    'exp': _bw_exp,
    'log': _bw_log,
    'sqrt': _bw_sqrt,
    'dot_general': _bw_dot_general,
    'reduce_sum': _bw_reduce_sum,
    'broadcast_in_dim': _bw_broadcast,
    'reshape': _bw_reshape,
    'squeeze': _bw_squeeze,
    'transpose': _bw_transpose,
    'slice': _bw_slice,
    'pad': _bw_pad,
    'concatenate': _bw_concat,
    'iota': _bw_iota,
}


# ---- driver ----------------------------------------------------------

def edge_push_correction(closed_jaxpr: ClosedJaxpr, x, r_value):
    """Σ_i r_value[i] · ∇²r_i(x).  Single forward + single backward sweep."""
    sj = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts

    if len(sj.invars) != 1:
        raise NotImplementedError("multi-input residual jaxpr (extend if needed)")
    invar = sj.invars[0]
    n = int(invar.aval.size)
    dtype = x.dtype

    # env: var -> (primal, jacobian or None)
    env = {}

    # Initialize constvars with zero Jacobians (they are constants wrt x).
    for cv, cval in zip(sj.constvars, consts):
        env[cv] = (cval, None)
    # Initialize invar with primal=x and J=identity (reshaped if needed).
    eye = jnp.eye(n, dtype=dtype).reshape(invar.aval.shape + (n,))
    env[invar] = (x, eye)

    # ---- forward sweep ----
    for eqn in sj.eqns:
        rule = FORWARD_RULES.get(eqn.primitive.name)
        if rule is None:
            raise NotImplementedError(f"forward rule for {eqn.primitive.name}")
        primals = [_val(env, v) for v in eqn.invars]
        jacs    = [_jac(env, v) for v in eqn.invars]
        out_p, out_j = rule(primals, jacs, eqn.params)
        # If single outvar (typical), bind it; multi-outvar primitives unsupported.
        assert len(eqn.outvars) == 1, eqn.primitive.name
        env[eqn.outvars[0]] = (out_p, out_j)

    # ---- backward sweep ----
    out_var = sj.outvars[0]
    cotangents = {out_var: jnp.asarray(r_value).reshape(out_var.aval.shape)}
    accum = jnp.zeros((n, n), dtype=dtype)

    for eqn in reversed(sj.eqns):
        out_var_e = eqn.outvars[0]
        if out_var_e not in cotangents:
            continue  # unused branch; skip
        cot_out = cotangents.pop(out_var_e)
        rule = BACKWARD_RULES.get(eqn.primitive.name)
        if rule is None:
            raise NotImplementedError(f"backward rule for {eqn.primitive.name}")
        in_cots, accum = rule(eqn, env, cot_out, accum)
        for v, ct in zip(eqn.invars, in_cots):
            if ct is None or isinstance(v, Literal):
                continue
            if v in cotangents:
                cotangents[v] = cotangents[v] + ct
            else:
                cotangents[v] = ct

    return accum


def edge_push_correction_fn(r_fn, x):
    """Drop-in replacement for `residual_correction_branch`.

    Signature matches `correction_fn=` accepted by `nls_hessian`.

    NOTE: needs the *closed jaxpr* of r, not just r_fn.  In the NLS API
    we have the closed jaxpr at hand (from the detector); a thin wrapper
    in `api.py` passes it through.  This function exists primarily for
    testing — production callers should go through `api.nls_hessian`.
    """
    closed_jaxpr = jax.make_jaxpr(r_fn)(x)
    r_value = r_fn(x)
    return edge_push_correction(closed_jaxpr, x, r_value)
