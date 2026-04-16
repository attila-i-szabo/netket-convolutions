"""Shared bits"""

from warnings import warn

import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
from einops import rearrange

# All layers defined here have kernels of shape [out_features, in_features, n_symm]
default_equivariant_initializer = lecun_normal(in_axis=1, out_axis=0)


def check_input_size(x: jnp.ndarray, n: int, strict: bool) -> jnp.ndarray:
    """Check if input feature maps are compatible with expected feature size.

    If `strict` is `False`, allow interpreting larger last dimension as
    sublattice structure, and reshuffle it as additional feature maps."""

    if strict:
        assert x.shape[-1] == n, f"Invalid feature map size {x.shape[-1]} != {n}"
    else:
        assert x.shape[-1] % n == 0, f"Invalid feature map size {x.shape[-1]} != {n}*N"
        if x.shape[-1] != n:
            warn("Oversized feature maps interpreted as extra sublattice features")
            x = rearrange(x, "... ft (cell sl) -> ... (ft sl) cell", cell=n)

    return x
