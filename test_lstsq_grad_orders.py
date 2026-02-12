"""Test lstsq gradient correctness at all orders, including reverse mode."""
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)

# Overdetermined system (m > n, non-zero residual → B-term active)
m, n = 6, 3
a = jax.random.normal(k1, (m, n))
b = jax.random.normal(k2, (m, 1))

def loss_a(a_):
    x, _, _, _ = jnp.linalg.lstsq(a_, b)
    return jnp.sum(x ** 2)

def loss_b(b_):
    x, _, _, _ = jnp.linalg.lstsq(a, b_)
    return jnp.sum(x ** 2)

# Finite differences for scalar-valued function
def fd_grad(f, x, eps=1e-6):
    """Central difference gradient of scalar function."""
    g = np.zeros_like(x)
    it = np.nditer(np.zeros(x.shape), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        e = np.zeros_like(x)
        e[idx] = eps
        e_jnp = jnp.array(e)
        fp = float(f(x + e_jnp))
        fm = float(f(x - e_jnp))
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return jnp.array(g)

# Finite differences for gradient-valued function (returns Hessian)
def fd_hessian(f, x, eps=1e-5):
    """Central difference Hessian via FD of gradient."""
    g_func = grad(f)
    shape = x.shape
    H = np.zeros(shape + shape)
    it = np.nditer(np.zeros(shape), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        e = np.zeros_like(x)
        e[idx] = eps
        e_jnp = jnp.array(e)
        gp = np.array(g_func(x + e_jnp))
        gm = np.array(g_func(x - e_jnp))
        fd_col = (gp - gm) / (2 * eps)
        H[(...,) + idx] = fd_col
        it.iternext()
    return jnp.array(H)

print("=" * 60)
print("1st-order gradients")
print("=" * 60)

g_a = grad(loss_a)(a)
g_a_fd = fd_grad(loss_a, a)
err1_a = float(jnp.max(jnp.abs(g_a - g_a_fd)))
print(f"  grad(loss_a) vs FD: max err = {err1_a:.2e}")

g_b = grad(loss_b)(b)
g_b_fd = fd_grad(loss_b, b)
err1_b = float(jnp.max(jnp.abs(g_b - g_b_fd)))
print(f"  grad(loss_b) vs FD: max err = {err1_b:.2e}")

print()
print("=" * 60)
print("2nd-order gradients w.r.t. a (overdetermined, B-term active)")
print("=" * 60)

# Pure forward-forward
H_ff = jacfwd(jacfwd(loss_a))(a)
print("  jacfwd(jacfwd) done")
# Forward-reverse (forward over grad)
H_fr = jacfwd(grad(loss_a))(a)
print("  jacfwd(grad) done")
# Reverse-reverse (reverse over grad)
H_rr = jacrev(grad(loss_a))(a)
print("  jacrev(grad) done")
# FD of grad
H_fd = fd_hessian(loss_a, a)
print("  FD hessian done")

err_ff_fd = float(jnp.max(jnp.abs(H_ff - H_fd)))
err_fr_fd = float(jnp.max(jnp.abs(H_fr - H_fd)))
err_rr_fd = float(jnp.max(jnp.abs(H_rr - H_fd)))
err_ff_fr = float(jnp.max(jnp.abs(H_ff - H_fr)))

print(f"  jacfwd(jacfwd) vs FD:           {err_ff_fd:.2e}")
print(f"  jacfwd(grad)   vs FD:           {err_fr_fd:.2e}")
print(f"  jacrev(grad)   vs FD:           {err_rr_fd:.2e}")
print(f"  jacfwd(jacfwd) vs jacfwd(grad): {err_ff_fr:.2e}")

print()
print("=" * 60)
print("2nd-order gradients w.r.t. b")
print("=" * 60)

H_b_ff = jacfwd(jacfwd(loss_b))(b)
H_b_fr = jacfwd(grad(loss_b))(b)
H_b_fd = fd_hessian(loss_b, b)

err_b_ff_fd = float(jnp.max(jnp.abs(H_b_ff - H_b_fd)))
err_b_fr_fd = float(jnp.max(jnp.abs(H_b_fr - H_b_fd)))

print(f"  jacfwd(jacfwd) vs FD:  {err_b_ff_fd:.2e}")
print(f"  jacfwd(grad)   vs FD:  {err_b_fr_fd:.2e}")

print()
print("=" * 60)
print("Square system (zero residual, no B-term)")
print("=" * 60)

a_sq = jax.random.normal(k1, (n, n))
b_sq = jax.random.normal(k2, (n, 1))

def loss_sq(a_):
    x, _, _, _ = jnp.linalg.lstsq(a_, b_sq)
    return jnp.sum(x ** 2)

H_sq_ff = jacfwd(jacfwd(loss_sq))(a_sq)
H_sq_fr = jacfwd(grad(loss_sq))(a_sq)
H_sq_fd = fd_hessian(loss_sq, a_sq)

err_sq_ff_fd = float(jnp.max(jnp.abs(H_sq_ff - H_sq_fd)))
err_sq_fr_fd = float(jnp.max(jnp.abs(H_sq_fr - H_sq_fd)))

print(f"  jacfwd(jacfwd) vs FD:  {err_sq_ff_fd:.2e}")
print(f"  jacfwd(grad)   vs FD:  {err_sq_fr_fd:.2e}")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
threshold = 1e-4
all_ok = True
for name, err in [
    ("1st-order grad(a)", err1_a),
    ("1st-order grad(b)", err1_b),
    ("2nd-order ff(a) vs FD", err_ff_fd),
    ("2nd-order fr(a) vs FD", err_fr_fd),
    ("2nd-order rr(a) vs FD", err_rr_fd),
    ("2nd-order ff(b) vs FD", err_b_ff_fd),
    ("2nd-order fr(b) vs FD", err_b_fr_fd),
    ("2nd-order sq ff vs FD", err_sq_ff_fd),
    ("2nd-order sq fr vs FD", err_sq_fr_fd),
]:
    status = "PASS" if err < threshold else "FAIL"
    if err >= threshold:
        all_ok = False
    print(f"  [{status}] {name}: {err:.2e}")

print()
if all_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
