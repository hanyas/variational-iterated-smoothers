# Docstring style

`varsmooth` uses a single hybrid docstring convention: a NumPy-like layout
under Google-like section headers. Apply it to every public function, class,
and method.

## Rules

1. **Section headers** are `Args:`, `Returns:`, `Raises:`, and `Attributes:`
   (for `NamedTuple`s), each followed by a newline. Do **not** use NumPy dash
   underlines (`----------`) and do **not** use a `Parameters` header.

2. **Each entry is `name: Type` on its own line**, with the description on the
   following line(s), indented one level deeper:

   ```
   Args:
       model: AdditiveGaussianModel
           The state-space model to linearize.
       damping: float
           Trust-region damping in [0, 1); related to temperature t by
           damping = t / (1 + t).

   Returns:
       posterior: GaussMarkov
           The updated forward Gauss-Markov posterior.
       feasible_flags: Array
           Boolean array of shape (T,) marking feasible steps.
   ```

   Prefer **named** return values when a function returns a tuple. Otherwise
   use a single unnamed `Returns:` block in the same layout (`Type` on the
   first line, description indented below).

3. **No double-backtick markup.** Write identifiers, math, and shapes as plain
   text: `-0.5 x^T M x + v^T x + c`, `shape (T, dx)`, etc. Unicode
   superscripts/subscripts already present may stay, but do not add new markup.

4. **Every public function, class, and method** gets at least a one-line
   imperative summary ending in a period. Multi-argument public functions get
   full `Args:`/`Returns:` blocks. Private helpers (leading underscore) get a
   one-line summary only when their purpose is non-obvious; no boilerplate.

5. **NamedTuples** get an `Attributes:` block in the same `name: Type` +
   indented-description layout.

6. **Shape conventions** are stated once in the module docstring of
   `varsmooth/objects.py` (`dx` state dim, `dy` observation dim, `T`
   transitions, so a trajectory has `T + 1` marginals and `T`
   kernels/observations). Reference them elsewhere rather than re-explaining.

7. **Paper notation.** Where the notation follows the paper
   (arXiv:2511.15409), say so. The quadratic-potential `NamedTuple`s in
   `objects.py` are mapped to the paper's symbols in a module-level table.

## Full example

```python
def line_search(
    init_param: float,
    fun: Callable,
    grad: Callable,
    rtol: float = 0.1,
) -> tuple[float, float, float, bool]:
    """Find a temperature satisfying the trust-region constraint by bisection.

    Args:
        init_param: float
            Initial temperature at which to start the search.
        fun: Callable
            Dual objective temperature -> value; one full message pass.
        grad: Callable
            Constraint slack temperature -> (kl_constraint - realized_KL).
        rtol: float
            Absolute tolerance on the slack magnitude for termination.

    Returns:
        param: float
            The accepted temperature.
        fn_val: float
            The dual objective at param.
        slack: float
            The constraint slack at param.
        feasible: bool
            Whether a feasible temperature was found.
    """
    ...


class ValueFn(NamedTuple):
    """Quadratic value function with fields (R, r, rho).

    Attributes:
        R: Array
            Quadratic-form matrix of shape (dx, dx).
        r: Array
            Linear coefficient vector of shape (dx,).
        rho: Array
            Scalar constant offset.
    """

    R: Array
    r: Array
    rho: Array
```
