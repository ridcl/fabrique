import logging
from dataclasses import dataclass
from typing import Dict, List

import jax
import jax.numpy as jnp
from flax import nnx
from jinja2 import Environment
from multimethod import multimethod

from fabrique.generation import sample
from fabrique.loading import from_pretrained

logger = logging.getLogger("fabrique")


@dataclass
class ChatMessage:
    role: str
    content: str

    @staticmethod
    def from_response(response: str):
        role_content = response.split("\n\n", 1)
        if len(role_content) != 2:
            raise ValueError(
                "Couldn't parse LLM response into a chat message. "
                + f"Response text:\n\n{response}"
            )
        else:
            role, content = role_content
            return ChatMessage(role=role, content=content)

    @staticmethod
    def from_dict(d: Dict):
        return ChatMessage(**d)

    def dict(self):
        return {"role": self.role, "content": self.content}


class SpecialTokenMap:

    def __init__(self, hf_config: Dict):
        # BOS
        self.bos_id = hf_config["bos_token_id"]
        # PAD, may not present
        self.pad_id = hf_config.get("pad_token_id")
        # EOS, may be a single int or a tuple
        self.eos_ids = hf_config["eos_token_id"]
        if isinstance(self.eos_ids, int):
            self.eos_ids = (self.eos_ids,)
        elif isinstance(self.eos_ids, list):
            self.eos_ids = tuple(self.eos_ids)
        self.eos_id = self.eos_ids[0]


class LLM:

    def __init__(self, tokenizer, model, hf_config: dict, name: str = ""):
        self.tokenizer = tokenizer
        self.model = model
        self.hf_config = hf_config
        self.name = name
        self.rngs = nnx.Rngs(0)
        self.special_tokens = SpecialTokenMap(hf_config)
        self.chat_template = None
        if tmpl := self.hf_config.get("chat_template"):
            self.chat_template = Environment().from_string(tmpl)

    def __repr__(self):
        return f'LLM("{self.name}")'

    @staticmethod
    def from_pretrained(repo_id: str, revision: str | None = None, **model_args):
        tokenizer, model, hf_config = from_pretrained(
            repo_id, revision=revision, **model_args
        )
        return LLM(tokenizer, model, hf_config, name=repo_id)

    def apply_chat_template(self, messages: List[ChatMessage]):
        msg_dicts = [msg.dict() for msg in messages]
        assert self.chat_template is not None, "This LLM doesn't have a chate template"
        return self.chat_template.render(
            messages=msg_dicts,
            bos_token=self.tokenizer.id_to_token(self.special_tokens.bos_id),
            # eos_token=self.tokenizer.id_to_token(self.eos_token_id),
        ).strip()

    def collate_with_padding(
        self, prompt_token_list: List[List[int]], pad_token_id: int
    ):
        if len(prompt_token_list) > 1:
            logger.warning(
                "Padding sequences from the beginning. Results for shorter "
                + "sequences may or may not be meaningful depending on the model."
            )
        max_tokens = max(len(tokens) for tokens in prompt_token_list)
        return jnp.array(
            [
                [pad_token_id] * (max_tokens - len(tokens)) + tokens
                for tokens in prompt_token_list
            ]
        )

    @multimethod
    def generate(
        self,
        prompts: List[str],
        new_only: bool = True,
        max_length: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        prng_key: jax.Array | None = None,
        skip_special_tokens: bool = True,
    ):
        max_batch_size = self.model.args.max_batch_size
        if len(prompts) > max_batch_size:
            raise ValueError(
                f"Trying to generate response for {len(prompts)} prompts, "
                + f"but model is initialized with max_batch_size = {max_batch_size}"
            )
        prompt_token_list = [enc.ids for enc in self.tokenizer.encode_batch(prompts)]
        prompt_tokens = self.collate_with_padding(
            prompt_token_list, self.special_tokens.pad_id or self.special_tokens.bos_id
        )
        sequences = sample(
            self.model,
            prompt_tokens,
            pad_token_id=self.special_tokens.pad_id or self.special_tokens.eos_id,
            eos_token_id=self.special_tokens.eos_ids,  # note: eos_ids, plural
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prng_key=prng_key if prng_key is not None else self.rngs(),
        )
        out = []
        for seq in sequences:
            if new_only:
                seq = seq[prompt_tokens.shape[1] :]
            generated = self.tokenizer.decode(
                seq, skip_special_tokens=skip_special_tokens
            )
            out.append(generated)
        return out

    @multimethod  # type: ignore[no-redef]
    def generate(self, prompt: str, **kwargs):  # type: ignore[no-redef]
        return self.generate([prompt], **kwargs)[0]

    @multimethod  # type: ignore[no-redef]
    def generate(self, chats: List[List[ChatMessage]], **kwargs):
        prompts = [self.apply_chat_template(messages) for messages in chats]
        responses = self.generate(prompts, **kwargs)
        return [ChatMessage.from_response(response) for response in responses]

    @multimethod  # type: ignore[no-redef]
    def generate(self, messages: List[ChatMessage], **kwargs):
        return self.generate([messages], **kwargs)[0]
