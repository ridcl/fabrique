from collections.abc import Sequence
from functools import partial
from flax import nnx
import jax
from jax import numpy as jnp

# test imports

from gemma.multimodal import vision_utils as vu
from fabrique.loading import update_module_from_params
from fabrique.loading import LoadRule as R




class MlpBlock(nnx.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        block_id: int,
        input_dim: int,
        mlp_dim: int | None = None,  # Defaults to 4x input dim
        dropout: float = 0.0,
        dtype_mm: str = "float32",
        *,
        rngs: nnx.Rngs,
    ):
        self.block_id = block_id
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dtype_mm = dtype_mm
        linear = partial(
            nnx.Linear,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
            dtype=dtype_mm,
            rngs=rngs,
        )
        mlp_dim = self.mlp_dim or 4 * input_dim
        self.linear1 = linear(input_dim, mlp_dim)
        self.do = nnx.Dropout(rate=self.dropout, rngs=rngs)
        self.linear2 = linear(mlp_dim, input_dim)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Applies Transformer MlpBlock module."""
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.do(x, deterministic=deterministic)
        x = self.linear2(x)
        return x



def test_mlp_block():
    batch_size = 2
    length = 5
    input_dim = 4
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    mlp_nn = vu.MlpBlock(block_id=0, mlp_dim=mlp_dim)
    variables = mlp_nn.init(rngs.params(), x, deterministic=True)
    mlp = MlpBlock(block_id=0, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)
    rules = [
        R("Dense_0.kernel", "linear1.kernel"),
        R("Dense_0.bias", "linear1.bias"),
        R("Dense_1.kernel", "linear2.kernel"),
        R("Dense_1.bias", "linear2.bias"),
    ]
    update_module_from_params(mlp, rules, variables["params"])

    out_nn = mlp_nn.apply(variables, x)
    out = mlp(x)
    assert jnp.all(out_nn == out)



class MAPHead(nnx.Module):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        block_id: int,
        input_dim: int,
        mlp_dim: int | None = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        dtype = jnp.float32,
        buggy: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        self.block_id = block_id
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim or 4 * input_dim
        self.num_heads = num_heads
        self.buggy = buggy

        init_fn = nnx.initializers.xavier_uniform()
        self.probe = nnx.Param(
            init_fn(key=rngs.params(), shape=(1, 1, input_dim), dtype=dtype)
        )

        self.attn = nnx.MultiHeadAttention(
            num_heads=self.num_heads, in_features=input_dim, kernel_init=init_fn, decode=False, rngs=rngs
        )
        self.norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.mlp = MlpBlock(self.block_id, input_dim=input_dim, mlp_dim=self.mlp_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        n, l, d = x.shape  # pylint: disable=unused-variable
        # probe = self.param(
        #     "probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype
        # )
        probe = jnp.tile(self.probe, [n, 1, 1])

        x = self.attn(probe, x)

        y = self.norm(x)
        if self.buggy:
            x = y
        x = x + self.mlp(y)
        return x[:, 0]


def test_map_head():
    batch_size = 2
    length = 5
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    map_head_nn = vu.MAPHead(block_id=0, mlp_dim=mlp_dim)
    variables = map_head_nn.init(rngs.params(), x)

    map_head = MAPHead(block_id=0, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)

    rules = [
        # probe
        R("probe", "probe"),
        # attn
        R("MultiHeadDotProductAttention_0.query.kernel", "attn.query.kernel"),
        R("MultiHeadDotProductAttention_0.query.bias", "attn.query.bias"),
        R("MultiHeadDotProductAttention_0.key.kernel", "attn.key.kernel"),
        R("MultiHeadDotProductAttention_0.key.bias", "attn.key.bias"),
        R("MultiHeadDotProductAttention_0.value.kernel", "attn.value.kernel"),
        R("MultiHeadDotProductAttention_0.value.bias", "attn.value.bias"),
        R("MultiHeadDotProductAttention_0.out.kernel", "attn.out.kernel"),
        R("MultiHeadDotProductAttention_0.out.bias", "attn.out.bias"),
        # norm
        R("LayerNorm_0.scale", "norm.scale"),
        R("LayerNorm_0.bias", "norm.bias"),
        # mlp
        R("MlpBlock_0.Dense_0.kernel", "mlp.linear1.kernel"),
        R("MlpBlock_0.Dense_0.bias", "mlp.linear1.bias"),
        R("MlpBlock_0.Dense_1.kernel", "mlp.linear2.kernel"),
        R("MlpBlock_0.Dense_1.bias", "mlp.linear2.bias"),
    ]
    update_module_from_params(map_head, rules, variables["params"])

    out_nn = map_head_nn.apply(variables, x)
    out = map_head(x)
    assert jnp.all(out == out_nn)


class Encoder1DBlock(nnx.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    def __init__(
        self,
        block_id: int,
        input_dim: int,
        mlp_dim: int | None = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        dropout: float = 0.0,
        dtype_mm: str = "float32",
        *,
        rngs: nnx.Rngs
    ):
        self.block_id = block_id
        self.input_dim = input_dim
        mlp_dim = mlp_dim or 4 * input_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dtype_mm = dtype_mm

        self.pre_attn_norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            in_features=input_dim,
            num_heads=self.num_heads,
            kernel_init=nnx.initializers.xavier_uniform(),
            # deterministic=deterministic,  # TODO: use in __call__!!
            dtype=self.dtype_mm,
            decode=False,
            rngs=rngs,
        )
        self.post_attn_dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.post_attn_norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.mlp = MlpBlock(
            block_id=self.block_id,
            input_dim=self.input_dim,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            rngs=rngs
        )
        self.post_mlp_dropout = nnx.Dropout(rate=dropout)


    def __call__(
        self, x: jax.Array, deterministic: bool = True
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        # TODO: figure out how to replicate nn.with_logical_constraint in NNX
        # x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        y = self.pre_attn_norm(x)

        y = self.attn(y, y, deterministic=deterministic)
        # y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
        y = self.post_attn_dropout(y, deterministic=deterministic)
        x = x + y

        y = self.post_attn_norm(x)
        y = self.mlp(y, deterministic)
        # y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
        y = self.post_mlp_dropout(y, deterministic=deterministic)
        x = x + y
        # x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        return x


def test_encoder_1d_block():
    batch_size = 2
    length = 5
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    block_nn = vu.Encoder1DBlock(block_id=0, mlp_dim=mlp_dim)
    variables = block_nn.init(rngs.params(), x)
    block = Encoder1DBlock(block_id=0, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)

    rules = [
        # attn
        R("MultiHeadDotProductAttention_0.query.kernel", "attn.query.kernel"),
        R("MultiHeadDotProductAttention_0.query.bias", "attn.query.bias"),
        R("MultiHeadDotProductAttention_0.key.kernel", "attn.key.kernel"),
        R("MultiHeadDotProductAttention_0.key.bias", "attn.key.bias"),
        R("MultiHeadDotProductAttention_0.value.kernel", "attn.value.kernel"),
        R("MultiHeadDotProductAttention_0.value.bias", "attn.value.bias"),
        R("MultiHeadDotProductAttention_0.out.kernel", "attn.out.kernel"),
        R("MultiHeadDotProductAttention_0.out.bias", "attn.out.bias"),
        # mlp
        R("MlpBlock_0.Dense_0.kernel", "mlp.linear1.kernel"),
        R("MlpBlock_0.Dense_0.bias", "mlp.linear1.bias"),
        R("MlpBlock_0.Dense_1.kernel", "mlp.linear2.kernel"),
        R("MlpBlock_0.Dense_1.bias", "mlp.linear2.bias"),
        # pre-attn norm
        R("LayerNorm_0.scale", "pre_attn_norm.scale"),
        R("LayerNorm_0.bias", "pre_attn_norm.bias"),
        # post-attn norm
        R("LayerNorm_1.scale", "post_attn_norm.scale"),
        R("LayerNorm_1.bias", "post_attn_norm.bias"),
    ]
    update_module_from_params(block, rules, variables["params"])

    out_nn = block_nn.apply(variables, x)
    out = block(x)
    assert jnp.all(out_nn == out)


class Encoder(nnx.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        depth: int,
        input_dim: int,
        mlp_dim: int | None = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        dropout: float = 0.0,
        scan: bool = False,
        remat_policy: str = "nothing_saveable",
        dtype_mm: str = "float32",
        *,
        rngs: nnx.Rngs
    ):
        assert not scan, "Encoding using scan() is not supported"
        self.depth = depth
        self.input_dim = input_dim
        mlp_dim = mlp_dim or 4 * input_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scan = scan
        self.remat_policy = remat_policy
        self.dtype_mm = dtype_mm

        self.blocks = nnx.List([
            Encoder1DBlock(
                block_id=lyr,
                dtype_mm=self.dtype_mm,
                input_dim=self.input_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                rngs=rngs
            )
            for lyr in range(self.depth)
        ])
        self.norm = nnx.LayerNorm(input_dim, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        for block in self.blocks:
            x = block(x, deterministic)
        x: jax.Array = self.norm(x)
        return x



def test_encoder():
    batch_size = 2
    length = 5
    depth = 6
    input_dim = 24
    mlp_dim = 4 * input_dim
    rngs = nnx.Rngs(params=0, data=45, dropout=99)
    x = jax.random.normal(rngs.data(), (batch_size, length, input_dim))

    encoder_nn = vu.Encoder(depth=depth, mlp_dim=mlp_dim)
    variables = encoder_nn.init(rngs.params(), x)
    encoder = Encoder(depth=depth, input_dim=input_dim, mlp_dim=mlp_dim, rngs=rngs)

    rules = [
        # block: attn
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.query.kernel", "blocks.{n}.attn.query.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.query.bias", "blocks.{n}.attn.query.bias"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.key.kernel", "blocks.{n}.attn.key.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.key.bias", "blocks.{n}.attn.key.bias"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.value.kernel", "blocks.{n}.attn.value.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.value.bias", "blocks.{n}.attn.value.bias"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.out.kernel", "blocks.{n}.attn.out.kernel"),
        R("encoderblock_{n}.MultiHeadDotProductAttention_0.out.bias", "blocks.{n}.attn.out.bias"),
        # block: mlp
        R("encoderblock_{n}.MlpBlock_0.Dense_0.kernel", "blocks.{n}.mlp.linear1.kernel"),
        R("encoderblock_{n}.MlpBlock_0.Dense_0.bias", "blocks.{n}.mlp.linear1.bias"),
        R("encoderblock_{n}.MlpBlock_0.Dense_1.kernel", "blocks.{n}.mlp.linear2.kernel"),
        R("encoderblock_{n}.MlpBlock_0.Dense_1.bias", "blocks.{n}.mlp.linear2.bias"),
        # block: pre-attn norm
        R("encoderblock_{n}.LayerNorm_0.scale", "blocks.{n}.pre_attn_norm.scale"),
        R("encoderblock_{n}.LayerNorm_0.bias", "blocks.{n}.pre_attn_norm.bias"),
        # block: post-attn norm
        R("encoderblock_{n}.LayerNorm_1.scale", "blocks.{n}.post_attn_norm.scale"),
        R("encoderblock_{n}.LayerNorm_1.bias", "blocks.{n}.post_attn_norm.bias"),
        # norm
        R("encoder_norm.scale", "norm.scale"),
        R("encoder_norm.bias", "norm.bias"),
    ]
    update_module_from_params(encoder, rules, variables["params"])

    out_nn = encoder_nn.apply(variables, x)
    out = encoder(x)
    assert jnp.allclose(out_nn, out)


# class ViTModel(nn.Module):
#   """ViT model.

#   Attributes:
#     compression_type: The compression type.
#     width: The model dimension of the vision encoder.
#     mlp_dim: The hidden dimension in the ffw layers.
#     num_heads: The number of the heads.
#     depth: The number of the layers.
#     patch_size: The size to patchify images.
#     posemb: The position embedding type.
#     dropout: The dropout rate.
#     scan: Whether to scan the layers (layer stacking).
#     remat_policy: The remat policy.
#     dtype_mm: The dtype to convert the input to.
#     output_length: Number of soft tokens per image.
#   """

#   patch_size: Sequence[int] = (14, 14)
#   width: int = 1152
#   depth: int = 27
#   mlp_dim: int | None = 4304  # Defaults to 4x input dim
#   num_heads: int = 16
#   posemb: str = "learn"  # Can also be "sincos2d"
#   dropout: float = 0.0
#   scan: bool = False
#   # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
#   remat_policy: str = "nothing_saveable"
#   dtype_mm: str = "float32"

#   def _get_posemb(
#       self,
#       typ: str,
#       *,
#       seqshape: tuple[int, int],
#       width: int,
#       name: str,
#       dtype: jnp.dtype = jnp.float32,
#   ) -> typing.Float["B M D"]:
#     """Returns the position embedding."""
#     if typ == "learn":
#       return self.param(
#           name,
#           nn.initializers.normal(stddev=1 / np.sqrt(width)),
#           (1, np.prod(seqshape), width),
#           dtype,
#       )
#     elif typ == "sincos2d":
#       return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype)
#     else:
#       raise ValueError(f"Unknown posemb type: {typ}")

#   @nn.compact
#   def __call__(
#       self,
#       image: typing.Float["B N P D"],
#       *,
#       train: bool = False,
#   ):
#     image = jnp.asarray(image, self.dtype_mm)

#     # Patch extraction
#     x = nn.Conv(
#         self.width,
#         self.patch_size,
#         strides=self.patch_size,
#         padding="VALID",
#         name="embedding",
#         dtype=self.dtype_mm,
#     )(image)

#     n, h, w, c = x.shape
#     x = jnp.reshape(x, [n, h * w, c])

#     # Add posemb before adding extra token.
#     x = x + self._get_posemb(
#         self.posemb,
#         seqshape=(h, w),
#         width=c,
#         name="pos_embedding",
#         dtype=x.dtype,
#     )

#     n, l, c = x.shape  # pylint: disable=unused-variable
#     x = nn.Dropout(rate=self.dropout)(x, not train)

#     x = Encoder(
#         depth=self.depth,
#         mlp_dim=self.mlp_dim,
#         num_heads=self.num_heads,
#         dropout=self.dropout,
#         scan=self.scan,
#         remat_policy=self.remat_policy,
#         dtype_mm=self.dtype_mm,
#         name="Transformer",
#     )(x, deterministic=not train)

#     return x
