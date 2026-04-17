"""Subroutines for converting masked to full kernels and for
decomposing GCNN kernels on space groups into regular convolutional
kernels acting on cosets of the translation subgroup."""

from typing import Callable

import numpy as np
import jax.numpy as jnp

from netket.utils.group import PermutationGroup
from netket.utils.types import Array, DType


def translation_table(shape: tuple[int]) -> Array:
    """Computes product/symmetry table of an n-dimensional translation group."""
    size = np.prod(shape)
    # product table of each 1d component
    ix = [(np.arange(x) - np.arange(x)[:, None]) % x for x in shape]
    ix = np.ix_(*[np.ravel(x) for x in ix])
    # product table with n_axes axes
    # each axis stands for row and column direction along one axis
    pt = np.arange(size).reshape(shape)[ix]
    shape = [x for x in shape for _ in range(2)]
    # separate row and column directions
    pt = pt.reshape(shape)
    # bring all row and all column directions together
    pt = pt.transpose(list(range(0, len(shape), 2)) + list(range(1, len(shape), 2)))

    return pt.reshape(size, size)


def expanded_index(permutation: Array, shape: tuple[int]) -> np.ndarray:
    """Computes indices that map (n_input) dimension of kernels to
    (output_per_cell, input_per_cell, *shape) as used in FFT-based group convolution.

    Args:
        permutation: (n_output, n_input) integer array, containing the
            permutation group or the product table.

            Both axes are assumed to decompose into (*shape, per_cell) dimensions
            going from major to minor.
        shape: Order of the translation group along each axis.

            Product of entries must divide both n_input, n_output.
    """
    n_cells = np.prod(np.asarray(shape))
    n_output, n_input = permutation.shape
    assert n_input % n_cells == 0, "Number of cells must divide input dimension"
    assert n_output % n_cells == 0, "Number of cells must divide output dimension"
    input_per_cell = n_input // n_cells
    output_per_cell = n_output // n_cells

    return (
        permutation[:, :input_per_cell]
        .reshape(n_cells, output_per_cell, input_per_cell)
        .transpose(1, 2, 0)
        .reshape(output_per_cell, input_per_cell, *shape)
    )


def unmask(
    permutation: PermutationGroup | Array | None,
    shape: tuple[int] | None,
    mask: Array | None,
) -> tuple[Callable[[Array], Array], int, int]:
    """Constructs a function that restores a masked kernel to its full size.

    Args:
        permutation: (n_output, n_input) integer array, containing the
            permutation group or the product table.

            Can also be `None` for simple convolutions.
        shape: Order of the translation group along each axis.
        mask: (n_input,) boolean array indicating nonzero entries of the kernel,
            or `None` for all-to-all kernels.

    Returns:
        unmask: function that builds the full kernel
        kernel_size: required last dimension of the masked kernel
        n_input: inferred input dimension
    """
    n_input = (
        np.prod(shape) if permutation is None else np.asarray(permutation).shape[1]
    )

    if mask is None:
        return (lambda x: x), n_input, n_input
    else:
        assert mask.shape == (n_input,), f"Expected mask of size {n_input}"
        (indices,) = np.nonzero(mask)  # convert mask to list of indices

        def unmask(kernel: Array) -> Array:
            kernel_full = jnp.zeros((*kernel.shape[:-1], n_input), kernel.dtype)
            kernel_full = kernel_full.at[..., indices].set(kernel)
            return kernel_full

        return unmask, len(indices), n_input


def expand_full(
    permutation: PermutationGroup | Array | None,
    shape: tuple[int],
    mask: Array | None,
) -> tuple[Callable[[Array], Array], int, int]:
    """Constructs a function that, given a (potentially masked) kernel,
    constructs the full kernel, expanded into translation group cosets.

    The input kernel is expected to have shape
        (output_features, input_features, n_input)
    The output kernel has shape
        (output_features, input_features, output_per_cell, input_per_cell, *shape)
    (pairs of axes in brackets are fused)

    Args:
        permutation: (n_output, n_input) integer array, containing the
            permutation group or the product table.

            Can also be `None` for simple convolutions.
        shape: Order of the translation group along each axis.
        mask: (n_input,) boolean array indicating nonzero entries of the kernel,
            or `None` for all-to-all kernels.

    Returns:
        expand: function that builds the expanded kernel
        kernel_size: required last dimension of the masked kernel
        n_input: inferred input dimension
    """
    if permutation is not None:
        permutation = np.asarray(permutation)  # in case we got a PermutationGroup
        index = expanded_index(permutation, shape)

    unmask_, kernel_size, n_input = unmask(permutation, shape, mask)

    if permutation is None:

        def expand(kernel: Array) -> Array:
            kernel = unmask_(kernel)
            return kernel.reshape(*kernel.shape[:-1], 1, 1, *shape)

    else:

        def expand(kernel: Array) -> Array:
            kernel = unmask_(kernel)
            return kernel[..., index]

    return expand, kernel_size, n_input


def expand_clipped(
    permutation: PermutationGroup | Array | None,
    shape: tuple[int],
    mask: Array,
) -> tuple[Callable[[Array], Array], tuple[tuple[int]], int, int]:
    """Constructs a function that, given a masked kernel,
    constructs the full kernel, expanded into translation group cosets,
    clipped to the narrowest LAX convolution spec.

    The input kernel is expected to have shape
        (output_features, input_features, n_input)
    The output kernel has shape
        (output_features, input_features, output_per_cell, input_per_cell, *shape_clip)
    where shape_clip is the shape of the smallest rectangular block in which
    all the expanded kernels can be fit.

    Also returns the sequence of (left_pad, right_pad) tuples required for PBC
    convolutions, and the required size of the masked kernel.

    Args:
        permutation: (n_output, n_input) integer array, containing the
            permutation group or the product table.

            Can also be `None` for simple convolutions.
        shape: Order of the translation group along each axis.
        mask: (n_input,) boolean array indicating nonzero entries of the kernel.

    Returns:
        expand: function that builds the restricted expanded kernel
        padding: tuple of (left, right) padding tuples for `jnp.pad`
        kernel_size: required last dimension of the masked kernel
        n_input: inferred input dimension
    """
    # in case we got PermutationGroup or HashableArray
    if permutation is not None:
        permutation = np.asarray(permutation)

    # check size of mask
    n_input = np.prod(shape) if permutation is None else permutation.shape[1]
    assert mask.shape == (n_input,), f"Expected mask of size {n_input}"
    mask = np.asarray(mask, dtype=bool)

    if permutation is None:
        # dummy kernel mapping for sizing mask
        index = np.arange(np.prod(shape)).reshape(1, 1, *shape)
    else:
        index = expanded_index(permutation, shape)

    # work out necessary padding along each dimension
    exp_mask = mask[index]
    padding = [(0, 0)] * 2  # for feature/cell dimensions
    # aggregate by axis and work out necessary padding on either side
    for axis in range(2, index.ndim):
        L = exp_mask.shape[axis]
        by_axis = exp_mask.sum(axis=tuple(x for x in range(index.ndim) if x != axis))
        (by_axis,) = np.nonzero(by_axis)
        # rightmost row with positive coordinate < L/2 with kernel entries
        # -> how far window extends to the right
        right_pad = np.max(by_axis[by_axis < L // 2])
        # leftmost row with negative coordinate > -L/2 with kernel entries
        # -> how far window extends to the left
        left_pad = L - np.min(by_axis[by_axis >= L // 2])
        padding.append((left_pad, right_pad))

    # clipped index array
    # roll left padding area to front of each axis
    index = np.roll(index, [x[0] for x in padding], range(index.ndim))
    # clip dimensions above 2 to size
    index = index[:, :, *[slice(a + b + 1) for a, b in padding[2:]]]

    (mask_indices,) = np.nonzero(mask)  # convert mask to list of indices
    mask_in_index = np.expand_dims(index, -1) == mask_indices
    mask_in_index = np.where(
        np.any(mask_in_index, -1), np.argmax(mask_in_index, -1), mask_indices.size
    )  # invalid index for positions not under the mask

    def expand(kernel: Array) -> Array:
        kernel = jnp.take(kernel, mask_in_index, axis=-1, fill_value=0)
        return kernel

    return expand, tuple(padding), len(mask_indices), n_input
