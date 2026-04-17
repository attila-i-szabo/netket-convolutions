"""Group equivariant linear layers."""

from warnings import warn

import numpy as np
import jax.numpy as jnp

from jax import lax
from jax.nn.initializers import zeros
from flax.linen.module import Module, compact
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PrecisionLike

from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc

from ._base import default_equivariant_initializer, check_input_size
from . import _kernel_expand, _periodic_conv


class EquivariantMatrix(Module):
    r"""Implements a group convolution operation that is equivariant over a symmetry group
    by multiplying by the full kernel matrix"""

    features: int
    """The number of output features. Will be the second dimension of the output."""
    feature_group_count: int = 1
    """Number of feature groups for convolution.
    
    Must divide the number of both input and output features.
    For dense group convolution, should be 1 (default).
    For depthwise group convolution, should be the number of input features."""
    product_table: HashableArray | None = None
    """Product table for the space group."""
    shape: tuple[int] | None = None
    """Tuple that corresponds to shape of lattice, for simple convolutions.
    
    Ignored if `product_table` is given."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""
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

        # Size of output feature groups
        assert (
            self.features % self.feature_group_count == 0
        ), f"{self.feature_group_count = } must divide {self.features = }"
        self.features_per_group = self.features // self.feature_group_count

        # Parse product_table and/or shape
        if self.product_table is None:
            assert (
                self.shape is not None
            ), "Must supply either `product_table` or `shape`"
            self._product_table = _kernel_expand.translation_table(self.shape)
        else:
            if self.shape is not None:
                warn("EquivariantMatrix.shape is overridden by product_table")
            self._product_table = np.asarray(self.product_table)

        self.unmask, self.kernel_size, self.n_symm = _kernel_expand.unmask(
            self.product_table, self.shape, self.mask
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        x = check_input_size(x, self.n_symm, True)

        in_features = x.shape[-2]
        assert (
            in_features % self.feature_group_count == 0
        ), f"{self.feature_group_count = } must divide {in_features = }"
        in_features_per_group = in_features // self.feature_group_count

        # Separate feature groups and features within them
        x = x.reshape(
            *x.shape[:-2], self.feature_group_count, in_features_per_group, self.n_symm
        )

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features_per_group, self.kernel_size),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel, bias, x = promote_dtype(kernel, bias, x, dtype=None)

        kernel = self.unmask(kernel)
        kernel = kernel.reshape(
            self.feature_group_count,
            self.features_per_group,
            in_features_per_group,
            self.n_symm,
        )

        # Converts the convolutional kernel of shape (FGC, features, in_features, n_symm)
        # to a full dense kernel of shape (FGC, features, in_features, n_symm, n_symm)
        # result[group, out, in, g, h] == kernel[group, out, in, g^{-1}h]
        # input dimensions are [group, in, g], output dimensions are [group, out, h]
        kernel = jnp.take(kernel, jnp.asarray(self.product_table), 3)

        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 2, x.ndim - 1), (2, 3)), ((x.ndim - 3), (0,))),
            precision=self.precision,
        )
        # move feature group dim to correct place and reunite feature dims
        x = jnp.moveaxis(x, 0, -3)
        x = x.reshape(*x.shape[:-3], self.features, self.n_symm)

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        return x


class EquivariantFFT(Module):
    r"""Implements group convolution using
    a fast fourier transform over the translation group.

    The group convolution can be written in terms of translational convolutions with
    symmetry transformed filters as described in
    `Cohen et. al <http://proceedings.mlr.press/v48/cohenc16.pdf>_`

    The translational convolutions are then implemented with Fast Fourier Transforms.
    """

    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple[int]
    """Tuple that corresponds to shape of lattice."""
    feature_group_count: int = 1
    """Number of feature groups for convolution.
    
    Must divide the number of both input and output features.
    For dense group convolution, should be 1 (default).
    For depthwise group convolution, should be the number of input features."""
    product_table: HashableArray | None = None
    """Product table for the space group.
    May be omitted for simple convolutions."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    force_full_fft: bool = False
    """Use full-size complex FFT even if input and kernel are both real.
        Makes the output complex."""
    precision: PrecisionLike = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        assert (
            self.features % self.feature_group_count == 0
        ), f"{self.feature_group_count = } must divide {self.features = }"

        self.expand, self.kernel_size, self.n_symm = _kernel_expand.expand_full(
            self.product_table, self.shape, self.mask
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        x = check_input_size(x, self.n_symm, True)

        in_features = x.shape[-2]
        assert (
            in_features % self.feature_group_count == 0
        ), f"{self.feature_group_count = } must divide {in_features = }"
        in_features_per_group = in_features // self.feature_group_count

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features_per_group, self.kernel_size),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel, bias, x = promote_dtype(kernel, bias, x, dtype=None)

        kernel = self.expand(kernel)

        x = _periodic_conv.conv_fft(
            x,
            kernel,
            feature_group_count=self.feature_group_count,
            force_full_fft=self.force_full_fft,
            precision=self.precision,
        )

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        return x


class EquivariantLAX(Module):
    r"""Implements group convolution using cuDNN convolutions.

    The group convolution can be written in terms of translational convolutions with
    symmetry transformed filters as described in
    `Cohen et. al <http://proceedings.mlr.press/v48/cohenc16.pdf>_`

    The convolutions are then implemented with cuDNN convolutions
    with periodic padding."""

    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple[int]
    """Tuple that corresponds to shape of lattice."""
    mask: HashableArray
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""
    feature_group_count: int = 1
    """Number of feature groups for convolution.
    
    Must divide the number of both input and output features.
    For dense group convolution, should be 1 (default).
    For depthwise group convolution, should be the number of input features."""
    product_table: HashableArray | None = None
    """Product table for the space group.
    May be omitted for simple convolutions."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    force_full_fft: bool = False
    """Use full-size complex FFT even if input and kernel are both real.
        Makes the output complex."""
    precision: PrecisionLike = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        assert (
            self.features % self.feature_group_count == 0
        ), f"{self.feature_group_count = } must divide {self.features = }"

        self.expand, self.padding, self.kernel_size, self.n_symm = (
            _kernel_expand.expand_clipped(self.product_table, self.shape, self.mask)
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        x = check_input_size(x, self.n_symm, True)

        in_features = x.shape[-2]
        assert (
            in_features % self.feature_group_count == 0
        ), f"{self.feature_group_count = } must divide {in_features = }"
        in_features_per_group = in_features // self.feature_group_count

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features_per_group, self.kernel_size),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel, bias, x = promote_dtype(kernel, bias, x, dtype=None)

        kernel = self.expand(kernel)

        x = _periodic_conv.conv_lax(
            x,
            kernel,
            self.shape,
            self.padding,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        return x
