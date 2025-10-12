import math

import jax
import jax.numpy as jnp


def encode_batch(
    tokenizer,
    prompts: list[str],
    pad_to_multiple_of: int | None = None,
    truncate: int | None = None,
) -> jax.Array:
    """
    Encode batched prompts.

    Args:
        tokenizer: Tokenizer to use
        prompts: List of prompt strings
        pad_to_multiple_of: Align sequence length to multiple of this argument
            to reduce number of re-compilations
        truncate: Truncate the input to this value

    Returns:
        tokenized_sequences: JAX array of shape (batch_size, max_seq_len) with token IDs
    """

    tokenized_sequences = []

    # Encode each prompt
    for prompt in prompts:
        # Tokenize prompt and completion separately
        prompt_tokens = tokenizer.encode(prompt, add_bos=True)

        tokenized_sequences.append(prompt_tokens)

    # Find maximum sequence length for padding
    max_len = max(len(seq) for seq in tokenized_sequences)
    if pad_to_multiple_of:
        # align to multiple of certain value to minimize re-compilation for every length
        max_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of

    # Pad sequences and masks
    padded_sequences = []

    for seq in tokenized_sequences:
        # Pad sequence with PAD tokens
        padding_length = max_len - len(seq)
        padded_seq = seq + [tokenizer.special_tokens.PAD] * padding_length

        padded_sequences.append(padded_seq)

    if truncate:
        padded_sequences = [seq[:truncate] for seq in padded_sequences]

    # Convert to JAX arrays
    tokenized_sequences = jnp.array(padded_sequences, dtype=jnp.int32)

    return tokenized_sequences


def encode_batch_for_prompt_completion(
    tokenizer,
    prompts: list[str],
    completions: list[str],
    pad_to_multiple_of: int | None = None,
    truncate: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Encode batched prompts and completions with completion masking.

    Args:
        prompts: List of prompt strings
        completions: List of completion strings (same length as prompts)

    Returns:
        Tuple of:
        * tokenized_sequences: JAX array of shape (batch_size, max_seq_len) with token IDs
        * completion_mask: JAX array of shape (batch_size, max_seq_len) with boolean mask
                          (True for completion tokens, False for prompt/padding tokens)
    """
    assert len(prompts) == len(
        completions
    ), "Prompts and completions must have same length"

    # batch_size = len(prompts)
    tokenized_sequences = []
    completion_masks = []

    # Encode each prompt + completion pair
    for prompt, completion in zip(prompts, completions):
        # Tokenize prompt and completion separately
        prompt_tokens = tokenizer.encode(prompt, add_bos=True)
        completion_tokens = tokenizer.encode(completion, add_eos=True)

        # Combine tokens
        full_sequence = prompt_tokens + completion_tokens

        # Create completion mask (False for prompt tokens, True for completion tokens)
        mask = [False] * len(prompt_tokens) + [True] * len(completion_tokens)

        tokenized_sequences.append(full_sequence)
        completion_masks.append(mask)

    # Find maximum sequence length for padding
    max_len = max(len(seq) for seq in tokenized_sequences)
    if pad_to_multiple_of:
        # align to multiple of certain value to minimize re-compilation for every length
        max_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of

    # Pad sequences and masks
    padded_sequences = []
    padded_masks = []

    for seq, mask in zip(tokenized_sequences, completion_masks):
        # Pad sequence with PAD tokens
        padding_length = max_len - len(seq)
        padded_seq = seq + [tokenizer.special_tokens.PAD] * padding_length

        # Pad mask with False (padding tokens are not completion tokens)
        padded_mask = mask + [False] * padding_length

        padded_sequences.append(padded_seq)
        padded_masks.append(padded_mask)

    if truncate:
        padded_sequences = [seq[:truncate] for seq in padded_sequences]
        padded_masks = [seq[:truncate] for seq in padded_sequences]

    # Convert to JAX arrays
    tokenized_sequences = jnp.array(padded_sequences, dtype=jnp.int32)
    completion_mask = jnp.array(padded_masks, dtype=jnp.bool_)

    return tokenized_sequences, completion_mask
