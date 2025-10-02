========
Sharding
========

.. note::
    This section is work in progress.

.. note::
    In the current implementation, all model parameters are first loaded to
    a single device and only then sharded between devices. This means that
    models that exceed single device memory currently cannot be loaded.
    However, if you managed to load the model and shard to multiple devices,
    JAX will automatically distribute activations, gradients and optimizer
    state during training.


Example of loading a model to multiple devices::

    import numpy as np
    import jax
    from jax.sharding import Mesh
    from fabrique.sampling import Sampler

    device_arr = np.array(jax.devices())[None, :]
    mesh = jax.sharding.Mesh(devices=device_arr, axis_names=("data", "model"))
    sampler = Sampler.load_model("gemma-3-4b-it", mesh=mesh)

To check that the model is indeed sharded, run::

    jax.debug.visualize_array_sharding(sampler.model.blocks[0].attn.q_einsum.kernel[:, :, 0])