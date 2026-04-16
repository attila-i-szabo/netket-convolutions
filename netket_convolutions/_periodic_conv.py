"""FFT and LAX convolutions with periodic boundary conditions."""

from typing import Any
import jax
import jax.numpy as jnp


def conv_fft(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    feature_group_count: int = 1,
    force_full_fft: bool = False,
    precision: Any = None,
) -> jnp.ndarray:
    """Performs an FFT-based convolution.

    Args:
        x: input of shape (..., in_features, in_feature_size)
        kernel: expanded convolutional kernel of shape
            (out_features, in_features_per_group, out_per_cell, in_per_cell, *shape)
        feature_group_count: Number of feature groups for convolution.
            Must divide the number of both input and output features.
        force_full_fft: Use full-size complex FFT even if both inputs are real.
            Makes the output complex.
    """
    out_features, in_features_per_group, out_per_cell, in_per_cell = kernel.shape[:4]
    shape = kernel.shape[4:]
    n_cells = jnp.prod(shape)
    batch_dims = x.shape[:-2]
    batch_size = jnp.prod(batch_dims)
    in_features, in_feature_size = x.shape[-2:]

    assert in_features == in_features_per_group * feature_group_count
    assert out_features % feature_group_count == 0
    out_features_per_group = out_features // feature_group_count
    assert in_feature_size == in_per_cell * n_cells
    out_feature_size = out_per_cell * n_cells

    x = x.reshape(
        batch_size, feature_group_count, in_features_per_group, n_cells, in_per_cell
    )
    x = jnp.moveaxis(x, -1, -2)
    x = x.reshape(*x.shape[:-1], *shape)

    kernel = kernel.reshape(
        feature_group_count, out_features_per_group, *kernel.shape[1:]
    )

    use_full_fft = jnp.iscomplexobj(x) or jnp.iscomplexobj(kernel) or force_full_fft

    if use_full_fft:
        fft_shape = shape
        n_fft = n_cells
        x = jnp.fft.fftn(x, s=shape).reshape(*x.shape[:4], n_fft)
        kernel = jnp.fft.fftn(kernel, s=shape).reshape(*kernel.shape[:5], n_fft)
    else:
        fft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        n_fft = jnp.prod(fft_shape)
        x = jnp.fft.rfftn(x, s=shape).reshape(*x.shape[:4], n_fft)
        kernel = jnp.fft.rfftn(kernel, s=shape).reshape(*kernel.shape[:5], n_fft)

    x = jax.lax.dot_general(
        x, kernel, (((2, 3), (2, 4)), ((1, 4), (0, 5))), precision=precision
    )
    x = x.transpose(2, 0, 3, 4, 1)
    x = x.reshape(*x.shape[:4], *fft_shape)

    if use_full_fft:
        x = jnp.fft.ifftn(x, s=shape).reshape(*x.shape[:4], n_cells)
    else:
        x = jnp.fft.irfftn(x, s=shape).reshape(*x.shape[:4], n_cells)

    x = jnp.moveaxis(x, -1, -2)
    x = x.reshape(*batch_dims, out_features, out_feature_size)

    return x


def conv_lax(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    padding: tuple[tuple[int]],
    feature_group_count: int = 1,
    precision: Any = None,
) -> jnp.ndarray:
    """Performs a periodically padded LAX convolution.

    Args:
        x: input of shape (..., in_features, in_feature_size)
        kernel: expanded convolutional kernel of shape
            (out_features, in_features_per_group, out_per_cell, in_per_cell, *shape)
        padding: left and right padding to be applied along each dimension
            before the convolution
        feature_group_count: Number of feature groups for convolution.
            Must divide the number of both input and output features.
    """
    out_features, in_features_per_group, out_per_cell, in_per_cell = kernel.shape[:4]
    shape = kernel.shape[4:]
    n_cells = jnp.prod(shape)
    batch_dims = x.shape[:-2]
    batch_size = jnp.prod(batch_dims)
    in_features, in_feature_size = x.shape[-2:]

    assert in_features == in_features_per_group * feature_group_count
    assert out_features % feature_group_count == 0
    assert in_feature_size == in_per_cell * n_cells
    out_feature_size = out_per_cell * n_cells

    x = x.reshape(batch_size, in_features, n_cells, in_per_cell)
    x = jnp.moveaxis(x, -1, -2)
    # fuse input feature and per-cell dimensions for cuDNN
    x = x.reshape(batch_size, in_features * in_per_cell, *shape)

    # fuse kernel feature and per-cell dimensions for cuDNN
    kernel = jnp.moveaxis(kernel, 1, 2)
    kernel = kernel.reshape(
        out_features * out_per_cell, in_features_per_group * in_per_cell, *shape
    )

    # pad input with periodic BC
    x = jnp.pad(x, padding, mode="wrap")
    # cuDNN convolution
    x = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1,) * len(shape),
        padding="VALID",
        feature_group_count=feature_group_count,
        precision=precision,
    )

    # split out_features and per-cell dimensions, fuse TG axes
    x = x.reshape(batch_size, out_features, out_per_cell, n_cells)
    x = jnp.moveaxis(x, -1, -2)
    x = x.reshape(*batch_dims, out_features, out_feature_size)

    return x
