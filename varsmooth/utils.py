import jax
import jax.numpy as jnp


def logdet(A):
    return jnp.linalg.slogdet(A)[1]


def symmetrize(A):
    return 0.5 * (A.T + A)


def none_or_idx(x, idx):
    if x is None:
        return None
    return jax.tree.map(lambda z: z[idx], x)


def none_or_shift(x, shift):
    if x is None:
        return None
    if shift > 0:
        return jax.tree.map(lambda z: z[shift:], x)
    return jax.tree.map(lambda z: z[:shift], x)


def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree.map(
            lambda a, b: jnp.concatenate([a[None, ...], b]), y, x
        )
    else:
        return jax.tree.map(
            lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x
        )


def bounded_while_loop(cond_fun, body_fun, init_val, maxiter):
    """``jax.lax.while_loop`` with an iteration cap.

    Iterates ``body_fun`` while ``cond_fun(val)`` holds, stopping after at most
    ``maxiter`` steps. Drop-in replacement for the previously used (and now
    archived) ``jaxopt._src.loop.while_loop`` with ``jit=True``.

    Args:
        cond_fun: predicate ``val -> bool`` controlling continuation.
        body_fun: update ``val -> val`` applied each iteration.
        init_val: initial loop-carried value (any pytree).
        maxiter: maximum number of iterations (static int).

    Returns:
        The final loop-carried value.
    """
    def _cond(carry):
        it, val = carry
        return jnp.logical_and(it < maxiter, cond_fun(val))

    def _body(carry):
        it, val = carry
        return it + 1, body_fun(val)

    _, val = jax.lax.while_loop(_cond, _body, (0, init_val))
    return val
