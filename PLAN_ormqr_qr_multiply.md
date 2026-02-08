# Implementation Plan: Expose `ormqr` / `qr_multiply` in JAX

**Issue:** [jax-ml/jax#29173](https://github.com/jax-ml/jax/issues/29173) — "Expose ormqr, allowing more efficient linear least squares solves using QR factorization"

## Problem Statement

JAX currently exposes `geqrf` (unpivoted QR factorization) and `geqp3` (column-pivoted QR factorization) which return Q in its compact Householder reflector form `(a_out, taus)`. To use Q, users must call `householder_product` (which wraps LAPACK `orgqr`/`ungqr`) to **materialize the full Q matrix**, then multiply via `@`.

This is wasteful for the common use case of computing `Q @ c` or `Q^H @ c` (e.g., in least-squares solves). LAPACK provides `ormqr` (real) / `unmqr` (complex) which apply Q directly from its compact Householder form without ever materializing the full matrix. This is both faster and more memory-efficient.

## Current State of the Codebase

### What exists today

| Layer | QR Factorization | Q Materialization | Q Application (missing) |
|-------|-----------------|-------------------|------------------------|
| **LAPACK** | `geqrf`, `geqp3` | `orgqr`/`ungqr` | `ormqr`/`unmqr` |
| **cuSolver** | `geqrf` | `orgqr` | `ormqr`/`unmqr` (available in cuSOLVER) |
| **JAX primitives** | `geqrf_p`, `geqp3_p` | `householder_product_p` | **NOTHING** |
| **jax.lax.linalg** | (internal) | `householder_product` | **NOTHING** |
| **jax.scipy.linalg** | `qr()` | `qr()` | **NOTHING** (scipy has `qr_multiply`) |
| **jax.numpy.linalg** | `qr(mode="raw")` | `qr()` | **NOTHING** |

### Key files

- `jax/_src/lax/linalg.py` — Primitives: `geqrf_p`, `geqp3_p`, `householder_product_p`, `qr_p`
- `jax/lax/linalg.py` — Public re-exports for `jax.lax.linalg`
- `jax/_src/scipy/linalg.py` — SciPy-compatible wrappers
- `jaxlib/cpu/lapack_kernels.h` — C++ LAPACK kernel declarations (template structs)
- `jaxlib/cpu/lapack_kernels.cc` — C++ FFI handler macros and implementations
- `jaxlib/cpu/lapack_kernels_using_lapack.cc` — LAPACK function pointer registration
- `jaxlib/cpu/lapack.cc` — Python module registration
- `jaxlib/gpu/solver_interface.h` — GPU solver interface declarations
- `jaxlib/gpu/solver_kernels_ffi.cc` — GPU FFI kernel implementations
- `tests/linalg_test.py` — Tests

---

## Implementation Plan

### Phase 1: CPU LAPACK FFI Binding for `ormqr`/`unmqr`

**Goal:** Register `ormqr` (real) and `unmqr` (complex) as FFI-callable kernels, following the exact same pattern as `OrthogonalQr` (orgqr/ungqr).

#### 1a. Declare the C++ template struct in `lapack_kernels.h`

Add a new template struct `OrthogonalQrMultiply` (analogous to `OrthogonalQr` at line 185):

```cpp
template <::xla::ffi::DataType dtype>
struct OrthogonalQrMultiply {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  // LAPACK signature: dormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
  using FnType = void(char* side, char* trans, lapack_int* m, lapack_int* n,
                      lapack_int* k, ValueType* a, lapack_int* lda,
                      ValueType* tau, ValueType* c, lapack_int* ldc,
                      ValueType* work, lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> a,       // Householder reflectors (from geqrf)
      ::xla::ffi::Buffer<dtype> tau,      // Householder scalars
      ::xla::ffi::Buffer<dtype> c,        // Matrix to multiply
      bool left,                          // side='L' (Q @ C) vs side='R' (C @ Q)
      bool transpose,                     // trans='N' vs trans='T'/'C'
      ::xla::ffi::ResultBuffer<dtype> c_out);

  static int64_t GetWorkspaceSize(char side, char trans, lapack_int m,
                                  lapack_int n, lapack_int k);
};
```

**Parameters explained:**
- `side`: 'L' for `Q @ C` (left multiply), 'R' for `C @ Q` (right multiply)
- `trans`: 'N' for Q, 'T' for Q^T (real), 'C' for Q^H (complex)
- `a`, `tau`: Output from `geqrf` (the compact Householder representation)
- `c`: The matrix to multiply by Q

#### 1b. Implement the kernel in `lapack_kernels.cc`

Implement `OrthogonalQrMultiply::Kernel` following the pattern of `OrthogonalQr::Kernel`. The kernel:
1. Copies `c` to `c_out` (in-place operation)
2. Queries workspace size with `lwork=-1`
3. Allocates workspace
4. Calls the LAPACK routine (`dormqr`/`sormqr`/`cunmqr`/`zunmqr`)
5. Handles batching by looping over batch dimensions

#### 1c. Define the FFI handler macro in `lapack_kernels.cc`

```cpp
#define JAX_CPU_DEFINE_ORMQR(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                          \
      name, OrthogonalQrMultiply<data_type>::Kernel,      \
      ::xla::ffi::Ffi::Bind()                             \
          .Arg<::xla::ffi::Buffer<data_type>>(/*a*/)      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*tau*/)    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*c*/)      \
          .Attr<bool>("left")                             \
          .Attr<bool>("transpose")                        \
          .Ret<::xla::ffi::Buffer<data_type>>(/*c_out*/))
```

Instantiate for all 4 types:
```cpp
JAX_CPU_DEFINE_ORMQR(lapack_sormqr_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_ORMQR(lapack_dormqr_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_ORMQR(lapack_cunmqr_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_ORMQR(lapack_zunmqr_ffi, ::xla::ffi::DataType::C128);
```

#### 1d. Register function pointers in `lapack_kernels_using_lapack.cc`

```cpp
jax::OrthogonalQrMultiply<ffi::DataType::F32>::FnType sormqr_;
jax::OrthogonalQrMultiply<ffi::DataType::F64>::FnType dormqr_;
jax::OrthogonalQrMultiply<ffi::DataType::C64>::FnType cunmqr_;
jax::OrthogonalQrMultiply<ffi::DataType::C128>::FnType zunmqr_;
```

And in the initialization:
```cpp
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32>>(sormqr_);
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64>>(dormqr_);
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64>>(cunmqr_);
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128>>(zunmqr_);
```

#### 1e. Register in `lapack.cc` (Python module)

Add the handler registrations in the `lapack.cc` Python module so they're accessible from Python.

---

### Phase 2: GPU cuSolver Binding for `ormqr`/`unmqr`

**Goal:** Add GPU support via cuSOLVER's `cusolverDn{S,D,C,Z}ormqr` / `unmqr`.

#### 2a. Add to `solver_interface.h`

```cpp
// Householder multiply: ormqr/unmqr
#define JAX_GPU_SOLVER_OrmqrBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, gpusolverSideMode_t side, \
  gpusolverOperation_t trans, int m, int n, int k
JAX_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, OrmqrBufferSize);
#undef JAX_GPU_SOLVER_OrmqrBufferSize_ARGS

#define JAX_GPU_SOLVER_Ormqr_ARGS(Type, ...)                                 \
  gpusolverDnHandle_t handle, gpusolverSideMode_t side,                      \
  gpusolverOperation_t trans, int m, int n, int k, Type *a, Type *tau,       \
  Type *c, int ldc, Type *workspace, int lwork, int *info
JAX_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Ormqr);
#undef JAX_GPU_SOLVER_Ormqr_ARGS
```

#### 2b. Implement in `solver_kernels_ffi.cc`

Follow the pattern of `OrgqrImpl` (lines 316-379):
- Template `OrmqrImpl` with side/trans parameters
- Dispatch by dtype
- Allocate workspace, loop over batch dimension
- Register FFI handler `OrmqrFfi`

**Note:** cuSOLVER provides `cusolverDnDormqr`/`cusolverDnSormqr` for real and `cusolverDnCunmqr`/`cusolverDnZunmqr` for complex. These are available since CUDA 10.1+.

---

### Phase 3: JAX Primitive — `ormqr_p`

**Goal:** Define the JAX primitive in `jax/_src/lax/linalg.py`, following the pattern of `householder_product_p`.

#### 3a. Public Python function

```python
def ormqr(a: ArrayLike, taus: ArrayLike, c: ArrayLike, *,
          left: bool = True, transpose: bool = False) -> Array:
  """Multiplies a matrix by Q from a QR factorization without materializing Q.

  Computes Q @ C (left=True, transpose=False), Q^T @ C (left=True, transpose=True),
  C @ Q (left=False, transpose=False), or C @ Q^T (left=False, transpose=True).

  For complex types, transpose=True computes the conjugate transpose (Q^H).

  Args:
    a: The lower-triangular Householder reflectors from geqrf/geqp3.
    taus: The Householder scalar factors from geqrf/geqp3.
    c: The matrix to multiply by Q.
    left: If True, compute Q @ C. If False, compute C @ Q.
    transpose: If True, use Q^T (or Q^H for complex types).

  Returns:
    The result of multiplying c by Q (or Q^T/Q^H).
  """
  return ormqr_p.bind(a, taus, c, left=left, transpose=transpose)
```

#### 3b. Shape and dtype rules

```python
def _ormqr_shape_rule(a_shape, taus_shape, c_shape, *, left, transpose):
  # Q is m x m (from m x n factorization with m >= n)
  # left=True:  Q @ C → C must be (m, p), result is (m, p)
  # left=False: C @ Q → C must be (p, m), result is (p, m)
  return c_shape

def _ormqr_dtype_rule(a_dtype, taus_dtype, c_dtype, **_):
  return c_dtype
```

#### 3c. Lowering rules

**CPU lowering:**
```python
def _ormqr_cpu_gpu_lowering(ctx, a, taus, c, *, left, transpose,
                             target_name_prefix):
  a_aval, _, _ = ctx.avals_in
  if target_name_prefix == "cpu":
    dtype = a_aval.dtype
    prefix = "un" if dtypes.issubdtype(dtype, np.complexfloating) else "or"
    target_name = lapack.prepare_lapack_call(f"{prefix}mqr_ffi", dtype)
  else:
    target_name = f"{target_name_prefix}solver_ormqr_ffi"
  rule = _linalg_ffi_lowering(target_name,
                               operand_output_aliases={2: 0})  # c aliased to output
  return rule(ctx, a, taus, c, left=left, transpose=transpose)
```

**Default/fallback lowering** (for TPU or platforms without native ormqr):
```python
def _ormqr_lowering(a, taus, c, *, left, transpose):
  # Fallback: materialize Q then multiply
  q = householder_product(a, taus)
  if transpose:
    q = _H(q)
  if left:
    return q @ c
  else:
    return c @ q
```

#### 3d. Register the primitive

```python
ormqr_p = standard_linalg_primitive(
    (_float | _complex, _float | _complex, _float | _complex), (2, 1, 2),
    _ormqr_shape_rule, "ormqr")
mlir.register_lowering(ormqr_p, mlir.lower_fun(_ormqr_lowering))
register_cpu_gpu_lowering(ormqr_p, _ormqr_cpu_gpu_lowering)
```

#### 3e. JVP rule

The JVP of `ormqr` is needed for autodiff through least-squares solves. This can be derived from the QR JVP rule. Consider:
- If `f(a, taus, c) = Q(a, taus) @ c`, then `df = dQ @ c + Q @ dc`
- `dQ` is related to the JVP of the QR factorization

For the initial implementation, a fallback JVP that materializes Q may be acceptable, with optimization deferred.

---

### Phase 4: Public API — `jax.lax.linalg.ormqr`

**Goal:** Export the new primitive through the public `jax.lax.linalg` namespace.

In `jax/lax/linalg.py`, add:
```python
from jax._src.lax.linalg import (
  ormqr as ormqr,
  ormqr_p as ormqr_p,
  ...
)
```

---

### Phase 5: SciPy-Compatible API — `jax.scipy.linalg.qr_multiply`

**Goal:** Provide `scipy.linalg.qr_multiply` in `jax.scipy.linalg`, mirroring SciPy's interface.

#### SciPy's API
```python
scipy.linalg.qr_multiply(a, c, mode='right', pivoting=False, conjugate=False)
```

- `mode='right'`: returns `(Q @ c, R)` — "right" means Q is on the right of the equation `A = QR`, so multiply Q on the left of c
- `mode='left'`: returns `(c @ Q^H, R)` — multiply Q^H on the right of c
- `pivoting=True`: uses column-pivoted QR (geqp3), returns `(result, R, P)`
- `conjugate=True`: uses `Q^H` instead of `Q`

**Note:** SciPy's `mode` naming is confusing. `mode='right'` means "the product Q @ c" (Q applied from the left), because "right" refers to the side of the decomposition. We should document this clearly.

#### JAX Implementation

```python
@implements(scipy.linalg.qr_multiply,
            lax_description="Only 'right' mode is supported currently.")
def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False):
  """Calculate the QR decomposition and multiply Q with a matrix.

  Args:
    a: array_like, shape (M, N). Matrix to be decomposed.
    c: array_like. Matrix to be multiplied by Q.
    mode: {'left', 'right'}, optional.
      Determines the order of multiplication:
      - 'right': returns (Q @ c, R)     [Q applied from left]
      - 'left':  returns (c @ Q^H, R)   [Q^H applied from right]
    pivoting: bool, optional. Use column-pivoted QR.
    conjugate: bool, optional. Use Q^H instead of Q.

  Returns:
    If pivoting is False: (result, R)
    If pivoting is True:  (result, R, P)
  """
  # 1. Compute the QR factorization (compact form)
  if pivoting:
    qr_result, taus = ...  # Use geqp3
  else:
    qr_result, taus = geqrf(a)

  # 2. Apply Q using ormqr (without materializing Q)
  if mode == 'right':
    result = lax_linalg.ormqr(qr_result, taus, c,
                               left=True, transpose=conjugate)
  elif mode == 'left':
    result = lax_linalg.ormqr(qr_result, taus, c,
                               left=False, transpose=(not conjugate))

  # 3. Extract R from qr_result
  r = jnp.triu(qr_result[..., :min(M, N), :])

  # Return
  if pivoting:
    return result, r, p
  return result, r
```

---

### Phase 6: Tests

Add tests in `tests/linalg_test.py`:

1. **`test_ormqr_basic`** — Verify `ormqr(a, taus, c, left=True)` equals `Q @ c` where Q is from `householder_product`
2. **`test_ormqr_transpose`** — Verify transpose/conjugate-transpose modes
3. **`test_ormqr_right`** — Verify `ormqr(a, taus, c, left=False)` equals `c @ Q`
4. **`test_ormqr_batched`** — Verify batched operation
5. **`test_ormqr_dtypes`** — Test all 4 dtypes: float32, float64, complex64, complex128
6. **`test_qr_multiply_scipy`** — Test `jax.scipy.linalg.qr_multiply` against `scipy.linalg.qr_multiply`
7. **`test_qr_multiply_pivoting`** — Test pivoted variant
8. **`test_ormqr_jvp`** — Test autodiff through ormqr

---

## Pivoted vs. Unpivoted Considerations

Both must be supported:

### Unpivoted QR (`geqrf` → `ormqr`)
- Standard use case: `A = QR`, solve `Rx = Q^T b`
- `ormqr` applies Q directly from geqrf output
- Straightforward — `ormqr` works directly with geqrf output

### Pivoted QR (`geqp3` → `ormqr`)
- Rank-revealing: `AP = QR` where P is a column permutation
- `ormqr` still works with geqp3 output (same Householder format)
- The permutation P must be tracked separately and applied to the solution
- The `qr_multiply` scipy API handles this with `pivoting=True`

The compact Householder form `(a_out, taus)` is the **same format** from both `geqrf` and `geqp3`, so the same `ormqr` primitive handles both. The only difference is that pivoted QR also returns permutation indices.

---

## Implementation Order and Dependencies

```
Phase 1 (CPU C++ FFI)  ──→  Phase 3 (JAX primitive)  ──→  Phase 4 (lax.linalg export)
                         ↗                              ↘
Phase 2 (GPU C++ FFI)  ─┘                                Phase 5 (scipy.linalg.qr_multiply)
                                                           ↓
                                                        Phase 6 (Tests)
```

**Minimum viable PR:** Phases 1, 3, 4, and 6 (CPU only, with fallback lowering for GPU/TPU)

**Full PR:** All 6 phases

---

## Open Questions & Risks

1. **GPU ormqr availability:** cuSOLVER has `cusolverDnDormqr` but it's not batched. May need a batched loop like orgqr. ROCM/HIP equivalents need verification.

2. **JVP complexity:** Differentiating through `ormqr` is non-trivial. A fallback that materializes Q for the JVP may be the pragmatic first step (matching how the QR JVP already works — it materializes Q).

3. **TPU support:** No native ormqr on TPU. The fallback lowering (materialize Q, then multiply) is needed and provides correctness without the performance benefit.

4. **Batching rule:** Need a `vmap`/batching rule for `ormqr_p`. This should follow the same pattern as `householder_product_p` — batch over the leading dimensions.

5. **Memory aliasing:** `ormqr` overwrites `c` in-place in LAPACK. The FFI binding needs `operand_output_aliases={2: 0}` to enable this optimization, with a copy for safety.

6. **SciPy API naming confusion:** SciPy's `mode='right'` means "Q applied from the left." JAX's lower-level `ormqr` should use `left=True/False` which is unambiguous. The scipy wrapper handles the translation.
