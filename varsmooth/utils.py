import jax
import jax.numpy as jnp


def logdet(A):
    """Return the log-determinant of A via slogdet, discarding the sign.

    Only the magnitude term of jnp.linalg.slogdet is returned, so the result
    is valid only for matrices with positive determinant. Every call site here
    passes a symmetric positive-definite matrix.

    Args:
        A: Array
            Square matrix of shape (n, n) assumed to have positive determinant.

    Returns:
        Array
            The scalar log|det A|.
    """
    return jnp.linalg.slogdet(A)[1]


def symmetrize(A):
    """Return the symmetric part 0.5 (A + A^T) of a square matrix."""
    return 0.5 * (A.T + A)


def none_or_idx(x, idx):
    """Index a pytree along its leading axis, passing None through unchanged.

    Args:
        x: Any
            A pytree whose leaves share a leading batch axis, or None.
        idx: int
            Index taken along the leading axis of every leaf.

    Returns:
        The indexed pytree, or None if x is None.
    """
    if x is None:
        return None
    return jax.tree.map(lambda z: z[idx], x)


def none_or_shift(x, shift):
    """Drop leading or trailing elements of a pytree, passing None through.

    Args:
        x: Any
            A pytree whose leaves share a leading batch axis, or None.
        shift: int
            If positive, drop the first shift elements (each leaf keeps
            z[shift:]); if negative, drop the last -shift elements (z[:shift]).

    Returns:
        The shifted pytree, or None if x is None.
    """
    if x is None:
        return None
    if shift > 0:
        return jax.tree.map(lambda z: z[shift:], x)
    return jax.tree.map(lambda z: z[:shift], x)


def none_or_concat(x, y, position=1):
    """Attach a single element y onto batched x, passing None through.

    Args:
        x: Any
            Batched pytree with a leading axis of length T, or None.
        y: Any
            A single element (one slice) to attach to x, or None.
        position: int
            1 to prepend y as the new first element, giving [y, *x]; -1 to
            append it as the new last element, giving [*x, y].

    Returns:
        The extended pytree of leading length T + 1, or None if either input
        is None.
    """
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree.map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree.map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)


def bounded_while_loop(cond_fun, body_fun, init_val, maxiter):
    """Run jax.lax.while_loop with a hard iteration cap.

    Args:
        cond_fun: Callable
            Predicate val -> bool controlling continuation.
        body_fun: Callable
            Update val -> val applied each iteration.
        init_val: Any
            Initial loop-carried value (any pytree).
        maxiter: int
            Maximum number of iterations (static int).

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
