"""Layers that may wrap any convolution implementation."""

from typing import Any, Literal

import numpy as np
import jax.numpy as jnp
from einops import rearrange

from jax import lax
from jax.nn.initializers import zeros, lecun_normal
from flax.linen.module import Module, compact
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PrecisionLike, DotGeneralT

from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc

from ._base import default_equivariant_initializer
from . import equivariant_linear, symmetric_linear


def DenseSymm(
    algorithm: Literal["FFT", "LAX", "matrix"],
    features: int,
    symmetries: HashableArray | None = None,
    shape: tuple[int] | None = None,
    use_bias: bool = True,
    mask: HashableArray | None = None,
    param_dtype: DType = jnp.float64,
    precision: PrecisionLike = None,
    kernel_init: NNInitFunc = default_equivariant_initializer,
    bias_init: NNInitFunc = zeros,
) -> Module:
    """General GCNN embedding layer.

    Args:
        algorithm: Convolution algorithm to use.
        features: The number of output features.
            Will be the second dimension of the output.
        symmetries: A group of symmetry operations or array of permutation indices
            over which the layer should be invariant.

            Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
            May be omitted for simple convolutions.
            Must be omitted if `algorithm="matrix"` and `shape` is supplied.
        shape: Tuple that corresponds to shape of lattice.

            Must be supplied unless `algorithm="matrix"` and `product_table` is given.
        use_bias: Whether to add a bias to the output (default: True)
        mask: Array of shape `(n_symm,)`, used to restrict the convolutional kernel.
            Only parameters with nonzero mask are used.

            For best performance, a boolean mask should be used.
        param_dtype: The dtype of the weights.
        precision: numerical precision of the computation,
            see :class:`jax.lax.Precision` for details.
        kernel_init: Initializer for the kernel. Defaults to Lecun normal.
        bias_init: Initializer for the bias. Defaults to zero initialization.
    """
    layer_type = {
        "FFT": symmetric_linear.DenseSymmFFT,
        "LAX": symmetric_linear.DenseSymmLAX,
        "matrix": symmetric_linear.DenseSymmMatrix,
    }[algorithm]
    return layer_type(
        features=features,
        symmetries=symmetries,
        shape=shape,
        use_bias=use_bias,
        mask=mask,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )


def Equivariant(
    algorithm: Literal["FFT", "LAX", "matrix"],
    features: int,
    feature_group_count: int = 1,
    product_table: HashableArray | None = None,
    shape: tuple[int] | None = None,
    use_bias: bool = True,
    mask: HashableArray | None = None,
    param_dtype: DType = jnp.float64,
    precision: PrecisionLike = None,
    kernel_init: NNInitFunc = default_equivariant_initializer,
    bias_init: NNInitFunc = zeros,
) -> Module:
    """General equivariant layer.

    Args:
        algorithm: Convolution algorithm to use.
        features: The number of output features.
            Will be the second dimension of the output.
        feature_group_count: Number of feature groups for convolution.

            Must divide the number of both input and output features.
            For dense group convolution, should be 1 (default).
            For depthwise group convolution, should be the number of input features.
        product_table: Product table for the space group.

            May be omitted for simple convolutions.
            Must be omitted if `algorithm="matrix"` and `shape` is supplied.
        shape: Tuple that corresponds to shape of lattice.

            Must be supplied unless `algorithm="matrix"` and `product_table` is given.
        use_bias: Whether to add a bias to the output (default: True)
        mask: Array of shape `(n_symm,)`, used to restrict the convolutional kernel.
            Only parameters with nonzero mask are used.

            For best performance, a boolean mask should be used.
        param_dtype: The dtype of the weights.
        precision: numerical precision of the computation,
            see :class:`jax.lax.Precision` for details.
        kernel_init: Initializer for the kernel. Defaults to Lecun normal.
        bias_init: Initializer for the bias. Defaults to zero initialization.
    """
    layer_type = {
        "FFT": equivariant_linear.EquivariantFFT,
        "LAX": equivariant_linear.EquivariantLAX,
        "matrix": equivariant_linear.EquivariantMatrix,
    }[algorithm]
    return layer_type(
        features=features,
        feature_group_count=feature_group_count,
        product_table=product_table,
        shape=shape,
        use_bias=use_bias,
        mask=mask,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )


class DensePenultimate(Module):
    """Dense linear layer acting on the penultimate dimension of the input."""

    features: int
    use_bias: bool = True
    param_dtype: DType = jnp.float64
    kernel_init: NNInitFunc = lecun_normal
    bias_init: NNInitFunc = zeros
    precision: PrecisionLike = None
    dot_general: DotGeneralT = lax.dot_general

    @compact
    def __call__(self, x: Array) -> Array:
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(x)[-2], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        x, kernel, bias = promote_dtype(x, kernel, bias)
        x = self.dot_general(
            x, kernel, (((x.ndim - 2,), (0,)), ((), ())), precision=self.precision
        )
        if bias is not None:
            x += bias
        x = jnp.moveaxis(x, -1, -2)
        return x


class MultiHeadEquivariant(Module):
    """Depthwise convolutional layer, where every feature within a single "head"
    share convolutional kernels."""

    algorithm: Literal["FFT", "LAX", "matrix"]
    """Convolution algorithm to use."""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    heads: int
    """The number of "attention heads", i.e., distinct convolutional kernels.
    
    Must divide `features`."""
    product_table: HashableArray | None = None
    """Product table for the space group.

    May be omitted for simple convolutions.
    Must be omitted if `algorithm="matrix"` and `shape` is supplied."""
    shape: tuple[int] | None = None
    """Tuple that corresponds to shape of lattice.
    
    Must be supplied unless `algorithm="matrix"` and `product_table` is given."""
    mix_heads: bool = True
    """Whether to add Dense layers before and after convolution to mix
    the different features (default: True)."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used.
        
    Must be given if `algorithm="LAX"`."""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: PrecisionLike = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    dense_kernel_init: NNInitFunc = lecun_normal
    """Initialiser for dense kernels if `mix_heads` is True."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        assert (
            self.features % self.heads == 0
        ), f"{self.heads = } must divide {self.features = }"
        self.d_eff = self.features // self.heads

        self.conv = Equivariant(
            algorithm=self.algorithm,
            product_table=self.product_table,
            features=self.heads,
            shape=self.shape,
            use_bias=False,
            mask=self.mask,
            feature_group_count=self.heads,  # fully depthwise
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
        )

        if self.mix_heads:
            self.before = DensePenultimate(
                features=self.features,
                use_bias=False,
                param_dtype=self.param_dtype,
                kernel_init=self.dense_kernel_init,
                precision=self.precision,
            )
            self.after = DensePenultimate(
                features=self.features,
                use_bias=False,
                param_dtype=self.param_dtype,
                kernel_init=self.dense_kernel_init,
                precision=self.precision,
            )

    @compact
    def __call__(self, x: Array) -> Array:
        if self.mix_heads:
            x = self.before(x)

        assert (
            x.shape[-2] == self.features
        ), f"Expected input with {self.features} feature maps"

        # split dimensions within and between heads
        x = rearrange(
            x, "batch (d_eff head) group -> (batch d_eff) head group", d_eff=self.d_eff
        )

        x = self.conv(x)

        # recombine feature dimensions
        x = rearrange(
            x, "(batch d_eff) head group -> batch (d_eff head) group", d_eff=self.d_eff
        )

        if self.mix_heads:
            x = self.after(x)

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features, 1), self.param_dtype
            )
            x += bias

        return x
