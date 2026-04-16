"""GCNN embedding linear layers."""

import numpy as np
import jax.numpy as jnp

from jax import lax
from jax.nn.initializers import zeros
from flax.linen.module import Module, compact
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PrecisionLike

from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc
from netket.errors import SymmModuleInvalidInputShape

from ._base import default_equivariant_initializer, check_input_size
from . import _kernel_expand, _periodic_conv


class DenseSymmMatrix(Module):
    r"""Implements a symmetrized linear transformation over a permutation group
    using matrix multiplication."""

    features: int
    """The number of output features. Will be the second dimension of the output."""
    symmetries: HashableArray | None = None
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.

        Must be omitted for simple convolutions if `shape` is supplied.
    """
    shape: tuple[int] | None = None
    """Tuple that corresponds to shape of lattice, for simple convolutions."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: PrecisionLike = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init

        # Parse symmetries and/or shape
        if self.symmetries is None:
            assert self.shape is not None, "Must supply either `symmetries` or `shape`"
            self._symmetries = _kernel_expand.translation_table(self.shape)
        else:
            assert self.shape is None, "Mustn't specify both `symmetries` and `shape`"
            self._symmetries = np.asarray(self.symmetries)

        self.n_symm, self.n_sites = self._symmetries.shape

        self.unmask, kernel_size = _kernel_expand.kernel_unmask(self.mask)
        self.kernel_size = kernel_size or self.n_sites

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the symmetrized linear transformation to the inputs along the last dimension.

        Args:
          x: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        # ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            raise SymmModuleInvalidInputShape("DenseSymmMatrix", x)

        x = check_input_size(x, self.n_sites, self.shape is None)

        in_features = x.shape[-2]

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.kernel_size),
            self.param_dtype,
        )

        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)

        kernel = self.unmask(kernel)

        # Converts the convolutional kernel of shape (self.features, in_features, n_sites)
        # to a full dense kernel of shape (self.features, in_features, n_symm, n_sites).
        # result[out, in, g, r] == kernel[out, in, g^{-1}r]
        kernel = jnp.take(kernel, jnp.asarray(self.symmetries), 2)

        # x is      (batches,       in_features,         n_sites)
        # kernel is (self.features, in_features, n_symm, n_sites)
        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 2, x.ndim - 1), (1, 3)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            # Convert symmetry-reduced bias of shape (features,) to the full bias of
            # shape (..., features, 1).
            x += jnp.expand_dims(bias, 1)

        return x


class DenseSymmFFT(Module):
    r"""Implements a symmetrized projection onto a space group using a Fast Fourier Transform"""

    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple
    """Tuple that corresponds to shape of lattice"""
    symmetries: HashableArray | None = None
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.

        May be omitted for simple convolutions.
    """
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    force_full_fft: bool = False
    """Use full-size complex FFT even if input and kernel are both real.
        Makes the output complex."""
    precision: PrecisionLike = None

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.expand, self.kernel_size = _kernel_expand.kernel_expand_full(
            self.symmetries, self.shape, self.mask, dtype=None
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the symmetrized linear transformation to the inputs along the last dimension.

        Args:
          x: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        # ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            raise SymmModuleInvalidInputShape("DenseSymmMatrix", x)

        x = check_input_size(x, self.n_sites, self.shape is None)

        in_features = x.shape[-2]

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.kernel_size),
            self.param_dtype,
        )

        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)

        kernel = self.expand(kernel)

        x = _periodic_conv.conv_fft(
            x, kernel, force_full_fft=self.force_full_fft, precision=self.precision
        )

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        return x


class DenseSymmLAX(Module):
    r"""Implements a symmetrized projection onto a space group using cuDNN convolutions."""

    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple
    """Tuple that corresponds to shape of lattice"""
    mask: HashableArray
    """Array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used"""
    symmetries: HashableArray | None = None
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.

        May be omitted for simple convolutions.
    """
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: PrecisionLike = None

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.expand, self.padding, self.kernel_size = (
            _kernel_expand.kernel_expand_clipped(
                self.symmetries, self.shape, self.mask, dtype=None
            )
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the symmetrized linear transformation to the inputs along the last dimension.

        Args:
          x: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        # ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            raise SymmModuleInvalidInputShape("DenseSymmMatrix", x)

        x = check_input_size(x, self.n_sites, self.shape is None)

        in_features = x.shape[-2]

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.kernel_size),
            self.param_dtype,
        )

        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)

        kernel = self.expand(kernel)

        x = _periodic_conv.conv_lax(x, kernel, self.padding, precision=self.precision)

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        return x
