from collections.abc import Sequence
from functools import partial
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


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
        self.do = nnx.Dropout(rate=self.dropout)
        self.linear2 = linear(mlp_dim, input_dim)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Applies Transformer MlpBlock module."""
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.do(x, deterministic=deterministic)
        x = self.linear2(x)
        return x


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
        self.post_attn_dropout = nnx.Dropout(rate=dropout)
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


class ViTModel(nnx.Module):
    # TODO: update docstring
    """ViT model.

    Attributes:
        compression_type: The compression type.
        width: The model dimension of the vision encoder.
        mlp_dim: The hidden dimension in the ffw layers.
        num_heads: The number of the heads.
        depth: The number of the layers.
        patch_size: The size to patchify images.
        posemb: The position embedding type.
        dropout: The dropout rate.
        remat_policy: The remat policy.
        dtype_mm: The dtype to convert the input to.
        output_length: Number of soft tokens per image.
    """

    def __init__(
        self,
        patch_size: Sequence[int] = (14, 14),
        in_channels: int = 3,
        # The name 'width' is unclear to me, but it corresponds to the number
        # of channels (image interpretation) or embedding dimension
        # (transformer interpretation)
        width: int = 1152,
        depth: int = 27,   # Number of encoder blocks
        mlp_dim: int | None = 4304,  # Defaults to 4x input dim
        num_heads: int = 16,
        posemb: str = "learn",  # Can also be "sincos2d"
        dropout: float = 0.0,
        scan: bool = False,
        # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
        remat_policy: str = "nothing_saveable",
        dtype_mm: str = "float32",
        *,
        rngs: nnx.Rngs
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.posemb = posemb
        self.dropout = dropout
        assert not scan, "scan() through layers is not supported"
        self.remat_policy = remat_policy
        self.dtype_mm = dtype_mm

        self.conv = nnx.Conv(
            self.in_channels,
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            dtype=self.dtype_mm,
            rngs=rngs,
        )
        init_fn = nnx.initializers.normal(stddev=1 / np.sqrt(width))
        # When using posemb == "learn", we must define embedding length
        # in advance. Here we use (64, 64) which corresponds to the Gemma's
        # default image size (896, 896) divided by patch size (14, 14).
        # Maybe we need to move it to model parameter, but assuming pre-trained
        # SigLip encoder, the use case for extra parameter is unclear.
        seqshape = (64, 64)
        self.pos_embedding = nnx.Param(
            init_fn(rngs.params(), shape=(1, np.prod(seqshape), width), dtype=dtype_mm)
        )
        self.do = nnx.Dropout(rate=self.dropout)
        self.encoder = Encoder(
            depth=self.depth,
            input_dim=self.width,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            rngs=rngs
        )

    def _get_posemb(
        self,
        typ: str,
        *,
        seqshape: tuple[int, int],
        width: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:    # typing.Float["B M D"]:
        """Returns the position embedding."""
        if typ == "learn":
            return self.pos_embedding
        elif typ == "sincos2d":
            return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype)
        else:
            raise ValueError(f"Unknown posemb type: {typ}")

    def __call__(
        self,
        image: jax.Array,   # typing.Float["B N P D"],
        *,
        train: bool = False,
    ):
        # TODO: assert (image.shape == (896, 896) or self.posemb != "learn") - incompatible
        image = jnp.asarray(image, self.dtype_mm)
        # Patch extraction
        x = self.conv(image)   # [bsz, n_patches, n_patches, width]
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        # Add posemb before adding extra token.
        x = x + self._get_posemb(
            self.posemb,
            seqshape=(h, w),
            width=c,
            dtype=x.dtype,
        )
        n, l, c = x.shape  # pylint: disable=unused-variable
        x = self.do(x, deterministic=not train)
        x = self.encoder(x, deterministic=not train)
        return x