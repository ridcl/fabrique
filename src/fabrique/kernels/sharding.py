"""Running custom (Pallas) kernels under data parallelism.

XLA cannot partition a Pallas kernel, so under a multi-device ``jit`` it
all-gathers the kernel's inputs and every device computes the *whole* batch.
Measured on 2 GPUs: 3.1x slower attention, and a gated-delta recurrence that
made the full training step 2x slower than the scan it replaced.  Wrapping the
call in ``shard_map`` over the batch axis makes each device process only its
own shard instead.

This needs the mesh to be known at trace time: set it with
``jax.set_mesh(mesh)`` (or ``jax.sharding.set_mesh``).  Without a mesh the
wrapper is a no-op, which is correct for a single device.
"""

from __future__ import annotations

import jax
from jax.sharding import PartitionSpec as P


def per_batch_shard(fn, n_args: int):
    """Wrap ``fn`` so it runs once per batch shard when a mesh is set.

    Every argument and output must have the batch as its leading axis, and the
    batch must be divisible by the number of devices in the mesh.  Outputs may
    be a pytree.  All mesh axes are treated as batch (data) axes.
    """
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return fn
    batch = P(mesh.axis_names)
    return jax.shard_map(
        fn,
        in_specs=(batch,) * n_args,
        out_specs=batch,
        # Pallas ops do not annotate how their outputs vary over mesh axes.
        check_vma=False,
    )
