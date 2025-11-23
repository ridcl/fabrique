import os
import logging

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from gemma import gm
from gemma.gm.ckpts._checkpoint import _CheckpointTree
from jax.sharding import NamedSharding

from fabrique.loading import LoadConfig, transform_tree, update_module_from_params
from fabrique.models.gemma.load_rules import RULES
from fabrique.models.gemma.modeling import Transformer
from fabrique.utils import ensure_path, set_by_path

logger = logging.getLogger("fabrique")


GEMMA_MODEL_MAP = {
    "gemma-3-1b-pt": (gm.nn.Gemma3_1B.config, gm.ckpts.CheckpointPath.GEMMA3_1B_PT),
    "gemma-3-1b-it": (gm.nn.Gemma3_1B.config, gm.ckpts.CheckpointPath.GEMMA3_1B_IT),
    "gemma-3-4b-pt": (gm.nn.Gemma3_4B.config, gm.ckpts.CheckpointPath.GEMMA3_4B_PT),
    "gemma-3-4b-it": (gm.nn.Gemma3_4B.config, gm.ckpts.CheckpointPath.GEMMA3_4B_IT),
    "gemma-3-12b-pt": (gm.nn.Gemma3_12B.config, gm.ckpts.CheckpointPath.GEMMA3_12B_PT),
    "gemma-3-12b-it": (gm.nn.Gemma3_12B.config, gm.ckpts.CheckpointPath.GEMMA3_12B_IT),
    "gemma-3-27b-pt": (gm.nn.Gemma3_27B.config, gm.ckpts.CheckpointPath.GEMMA3_27B_PT),
    "gemma-3-27b-it": (gm.nn.Gemma3_27B.config, gm.ckpts.CheckpointPath.GEMMA3_27B_IT),
}



def download_bucket_with_transfer_manager(
    bucket_name, destination_directory="", workers=8, max_results=1000
):
    """Download all of the blobs in a bucket, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to which to download all of the files. This
    # string is prepended (with os.path.join()) to the name of each blob to form
    # the full path. Relative paths and absolute paths are both accepted. An
    # empty string means "the current working directory". Note that this
    # parameter allows accepts directory traversal ("../" etc.) and is not
    # intended for unsanitized end user input.
    # destination_directory = ""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    # The maximum number of results to fetch from bucket.list_blobs(). This
    # sample code fetches all of the blobs up to max_results and queues them all
    # for download at once. Though they will still be executed in batches up to
    # the processes limit, queueing them all at once can be taxing on system
    # memory if buckets are very large. Adjust max_results as needed for your
    # system environment, or set it to None if you are sure the bucket is not
    # too large to hold in memory easily.
    # max_results=1000

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=max_results)]

    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))


def load_params_with_cache(ckpt):
    # if str(ckpt).startswith("gs://"):
    #     model_key = str(ckpt).split("/")[-1]
    #     local_path = os.path.expanduser(f"~/.cache/fabrique/models/{model_key}")
    #     if not os.path.exists(local_path):
    #         # Download original params
    #         params = gm.ckpts.load_params(ckpt)
    #         # Re-save them locally
    #         os.makedirs(os.path.pardir(local_path), exist_ok=True)
    #         ocp.StandardCheckpointer()
    # else:
    return gm.ckpts.load_params(ckpt)


def load_gemma_with_sharding(variant: str, *, mesh: jax.sharding.Mesh | None = None):
    raise AssertionError(
        "This implementation is currently broken, use `load_gemma` instead"
    )
    if variant not in GEMMA_MODEL_MAP:
        available_variants_str = "\n".join(f" - {v}" for v in GEMMA_MODEL_MAP.keys())
        raise ValueError(
            f"Model {variant} is not available. Available Gemma models are:\n{available_variants_str}"
        )
    config, ckpt_path = GEMMA_MODEL_MAP[variant]
    param_dtype = jnp.bfloat16
    model = nnx.eval_shape(
        lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    )

    graphdef, state = nnx.split(model)
    pspecs = nnx.get_partition_spec(state)
    pspecs_in_ckpt_layout = transform_tree(pspecs.raw_mapping, RULES, invert=True)
    if mesh is not None:
        sharding_in_ckpt_layout = jax.tree.map(
            lambda p: NamedSharding(mesh, p.value),
            pspecs_in_ckpt_layout,
            is_leaf=lambda p: isinstance(p, nnx.Param),
        )
        params_in_ckpt_layout = gm.ckpts.load_params(
            ckpt_path, sharding=_CheckpointTree(tree=sharding_in_ckpt_layout)
        )
    else:
        params_in_ckpt_layout = gm.ckpts.load_params(ckpt_path)
    params = transform_tree(params_in_ckpt_layout, RULES)
    # Above we ignored Rngs in vision encoder (ToNNX). Here we add it back
    ensure_path(params, "vision_encoder.rngs")
    set_by_path(params, "vision_encoder.rngs", nnx.state(nnx.Rngs(0)[1]).raw_mapping)
    # Re-create the model with actual params
    model = nnx.merge(graphdef, nnx.State(params))

    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model


def load_gemma(variant: str, *, mesh: jax.sharding.Mesh | None = None):
    if mesh is None:
        mesh = jax.make_mesh((1, jax.device_count()), ("data", "model"))
    if variant not in GEMMA_MODEL_MAP:
        available_variants_str = "\n".join(f" - {v}" for v in GEMMA_MODEL_MAP.keys())
        raise ValueError(
            f"Model {variant} is not available. Available Gemma models are:\n{available_variants_str}"
        )
    config, ckpt = GEMMA_MODEL_MAP[variant]
    param_dtype = jnp.bfloat16
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
        )
    params = load_params_with_cache(ckpt)
    update_module_from_params(model, RULES, params, mesh=mesh)
    # model.vision_encoder.rngs = nnx.Rngs(0)  # otherwise rngs will be abstract array
    tokenizer = gm.text.Gemma3Tokenizer()
    return tokenizer, model


LOAD_CONFIG = LoadConfig(
    model_re="gemma.*",
    loader=load_gemma,
)


#############################

# def main():
#     import jax
#     import numpy as np
#     from jax.sharding import NamedSharding

#     device_arr = np.array(jax.devices())[None, :]
#     mesh = jax.sharding.Mesh(devices=device_arr, axis_names=("data", "model"))
#     # variant = "gemma-3-12b-it"
#     variant = "gemma-3-4b-it"
#     t, m = load_gemma(variant, mesh=mesh)


#     config, ckpt_path = GEMMA_MODEL_MAP[variant]
#     param_dtype = jnp.bfloat16
#     model = nnx.eval_shape(
#         lambda: Transformer(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
#     )
#     PARAMS = gm.ckpts.load_params(ckpt_path)

#     state = nnx.state(model)
#     abstract_params = transform_tree(state.raw_mapping, RULES, invert=True)
#     abstract_arrays = transform_tree(state.raw_mapping, RULES, invert=True)

#     pspecs = transform_tree(nnx.get_partition_spec(state).raw_mapping, RULES, invert=True)  # strip out the annotations from state
#     pspecs_sharding = jax.tree.map(lambda p: NamedSharding(mesh, p.value), pspecs,
#                                    is_leaf=lambda p: isinstance(p, nnx.Param))

#     gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=pspecs))
#     gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=abstract_params))
#     gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=abstract_arrays))
#     params = gm.ckpts.load_params(ckpt_path, sharding=_CheckpointTree(tree=pspecs_sharding))


#     jax.lax.with_sharding_constraint(PARAMS, pspecs_sharding)
