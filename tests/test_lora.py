import jax
import jax.numpy as jnp
from flax import nnx
from fabrique import lora


def test__merge_lora_einsum_inplace():
    rngs = nnx.Rngs(0)
    einsum = nnx.Einsum("AB,BC->AC", kernel_shape=(10, 20), rngs=rngs)
    lora_einsum = lora.LoRAEinsum(5, einsum, rngs=rngs)
    # make lora_b also non-zero for this test
    lora_einsum.adapter.lora_b = jax.random.normal(rngs(), lora_einsum.adapter.lora_b.shape)
    X = jax.random.normal(rngs(), (2, 10))
    before_merge = lora_einsum(X)
    lora._merge_lora_einsum_inplace(lora_einsum)
    after_merge = lora_einsum(X)
    assert jnp.allclose(before_merge, after_merge, atol=1e-2)
    assert jnp.all(lora_einsum.adapter.lora_a == 0)
    assert jnp.all(lora_einsum.adapter.lora_b == 0)


def test_apply_merge():
    from fabrique.loading import load_model
    rngs = nnx.Rngs(89)
    x = jnp.arange(10)[None, :]
    _, model = load_model("gemma-3-1b-it")
    q_einsum_orig = model.blocks[4].attn.q_einsum
    out_orig = model(x)
    assert isinstance(q_einsum_orig, nnx.Einsum)

    lora.apply(model, 16, filter=lora.LORA_COMPATIBLE_MODULE, rngs=rngs)
    q_einsum_lora = model.blocks[4].attn.q_einsum
    out_lora = model(x)
    assert isinstance(q_einsum_lora, lora.LoRAEinsum)
    assert q_einsum_lora.base_module == q_einsum_orig
    assert jnp.all(out_orig.logits == out_lora.logits)

    # make non-zero
    q_einsum_lora.adapter.lora_b = jax.random.normal(rngs(), q_einsum_lora.adapter.lora_b.shape)
    out_lora_new = model(x)

    lora.merge(model)
    assert isinstance(model.blocks[4].attn.q_einsum, nnx.Einsum)
    out_new = model(x)
    assert jnp.all(out_lora_new.logits.argmax(-1) == out_new.logits.argmax(-1))
