import jax.numpy as jnp

from fabrique.loading import IGNORE
from fabrique.loading import ConversionRule as R

# fmt: off
RULES = [
    R("bert.embeddings.LayerNorm.gamma", "embeddings.norm.scale"),
    R("bert.embeddings.LayerNorm.beta", "embeddings.norm.bias"),
    R("bert.embeddings.position_embeddings.weight", "embeddings.position_embeddings.embedding"),
    R("bert.embeddings.word_embeddings.weight", "embeddings.token_embeddings.embedding"),
    R("bert.embeddings.token_type_embeddings.weight", "embeddings.segment_embeddings.embedding"),
    R("bert.encoder.layer.{n}.attention.output.LayerNorm.gamma", "layers[{n}].attention.norm.scale"),
    R("bert.encoder.layer.{n}.attention.output.LayerNorm.beta", "layers[{n}].attention.norm.bias"),
    R("bert.encoder.layer.{n}.attention.self.query.weight", "layers[{n}].attention.wq.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.self.query.bias", "layers[{n}].attention.wq.bias"),
    R("bert.encoder.layer.{n}.attention.self.key.weight", "layers[{n}].attention.wk.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.self.key.bias", "layers[{n}].attention.wk.bias"),
    R("bert.encoder.layer.{n}.attention.self.value.weight", "layers[{n}].attention.wv.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.self.value.bias", "layers[{n}].attention.wv.bias"),
    R("bert.encoder.layer.{n}.attention.output.dense.weight", "layers[{n}].attention.wo.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.output.dense.bias", "layers[{n}].attention.wo.bias"),

    R("bert.encoder.layer.{n}.intermediate.dense.weight", "layers[{n}].feed_forward.w1.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.intermediate.dense.bias", "layers[{n}].feed_forward.w1.bias"),
    R("bert.encoder.layer.{n}.output.dense.weight", "layers[{n}].feed_forward.w2.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.output.dense.bias", "layers[{n}].feed_forward.w2.bias"),
    R("bert.encoder.layer.{n}.output.LayerNorm.gamma", "layers[{n}].feed_forward.norm.scale"),
    R("bert.encoder.layer.{n}.output.LayerNorm.beta", "layers[{n}].feed_forward.norm.bias"),

    R("bert.pooler.dense.weight", "pooler.w.kernel", jnp.transpose),   # TODO: 1x768 vs 768x768
    R("bert.pooler.dense.bias", "pooler.w.bias"),

    R("cls.predictions.bias", IGNORE),
    R("cls.predictions.transform.LayerNorm.gamma", IGNORE),
    R("cls.predictions.transform.LayerNorm.beta", IGNORE),
    R("cls.seq_relationship.weight", IGNORE),
    R("cls.seq_relationship.bias", IGNORE),
    R("cls.predictions.transform.dense.weight", IGNORE),
    R("cls.predictions.transform.dense.bias", IGNORE),

]
# fmt: on
