import io

import jax
import jax.numpy as jnp
import requests
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor

from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl import params as param_lib


def main():
    # Define mesh
    MESH = [(1, 1), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

    # Load model params from Huggingface Hub
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    local_model_path = snapshot_download(repo_id=model_id, ignore_patterns=["*.pth"])
    processor = AutoProcessor.from_pretrained(model_id)

    # Load the model
    config = model_lib.ModelConfig.qwen3vl_4b()
    model = param_lib.create_model_from_safe_tensors(local_model_path, config, mesh)

    # Create a datapoint
    resp = requests.get(
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    img = Image.open(io.BytesIO(resp.content)).resize((448, 224))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Tokenize using HF processor
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    positions, rope_deltas = model_lib.get_rope_index(
        jnp.array(inputs.input_ids),
        jnp.array(inputs.image_grid_thw),
        None,
        input_mask=jnp.array(inputs.attention_mask),
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )
    grid_data = (
        model.visual.compute_grid_data(inputs.image_grid_thw)
        if "image_grid_thw" in inputs
        else None
    )

    out = model(
        jnp.array(inputs.input_ids),
        positions,
        jnp.array(inputs.pixel_values),
        grid_data,
        cache=None,
        attention_mask=jnp.array(inputs.attention_mask),
        output_hidden_states=True,
    )
    tokens = out[0].argmax(-1)
    last_token = tokens[0, -1]
    decoded_last_token = processor.decode(last_token.tolist())
    print(decoded_last_token)


if __name__ == "__main__":
    main()
