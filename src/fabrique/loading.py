import importlib
import json
import os
import pkgutil
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import jax
import safetensors.flax as st
from flax import nnx
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tqdm import tqdm

from fabrique import models
from fabrique.utils import set_nested_attr

###############################################################################
#                                  RULES                                      #
###############################################################################


@dataclass
class RuleIgnore:
    pass


IGNORE = RuleIgnore()


@dataclass
class ConversionRule:
    safe_pattern: str
    fab_pattern: str | RuleIgnore
    converter: Callable | None = None

    @property
    def safe_regexp(self):
        pat = self.safe_pattern
        pat = pat.replace(".", "\\.")
        pat = pat.replace("{n}", "(?P<n>\\d+)")
        pat = "^" + pat + "$"
        return re.compile(pat)


def maybe_apply_rule(rule: ConversionRule, safe_key: str, safe_val):
    if m := re.match(rule.safe_regexp, safe_key):
        if isinstance(rule.fab_pattern, RuleIgnore):
            return "", IGNORE
        path = rule.fab_pattern.format(**m.groupdict())
        val = rule.converter(safe_val) if rule.converter else safe_val
        return path, val
    else:
        return None


def convert_safetensor(rules: List[ConversionRule], safe_key: str, safe_val):
    for rule in rules:
        path_val = maybe_apply_rule(rule, safe_key, safe_val)
        if path_val:
            return path_val
    raise ValueError(
        f"Couldn't find a rule for safetensors key {safe_key} "
        + f"and value of shape {safe_val.shape}"
    )


def apply_rules(
    model: nnx.Module, rules: List[ConversionRule], flat: Dict[str, jax.Array]
):
    for safe_key, safe_val in flat.items():
        path, val = convert_safetensor(rules, safe_key, safe_val)
        if val is not IGNORE:
            fab_keys = path.split(".")
            fab_keys += ["value"]  # set to the .value field
            set_nested_attr(model, fab_keys, val)


###############################################################################
#                              MODEL UPDATE                                   #
###############################################################################


def update_model_from_safe(
    model: nnx.Module, rules: List[ConversionRule], model_dir: str
):
    """
    Update Flax NNX model from a Huggingface model directory
    """
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    model_file = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(index_file):
        with open(index_file) as fp:
            index = json.load(fp)
        safe_files_ = set(index["weight_map"].values())
        safe_files = [os.path.join(model_dir, filename) for filename in safe_files_]
    elif os.path.isfile(model_file):
        safe_files = [model_file]
    else:
        raise ValueError(f"Can't find safetensor files in {model_dir}")
    for path in tqdm(safe_files):
        flat = st.load_file(path)
        apply_rules(model, rules, flat)


###############################################################################
#                             HUGGINGFACE HUB                                 #
###############################################################################


@dataclass
class LoadConfig:
    model_types: List[str]
    model_args_class: type
    model_class: type
    rules: List[ConversionRule]
    chat_template: str | None = None


def find_load_configs():
    load_configs = {}
    for module_info in pkgutil.iter_modules(
        models.__path__, prefix=models.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        if hasattr(module, "LOAD_CONFIG"):
            cfg = getattr(module, "LOAD_CONFIG")
            for model_type in cfg.model_types:
                load_configs[model_type] = cfg
    return load_configs


def get_load_config(model_type: str):
    # collect load configs by traversing fabrique.models subpackages
    ret = find_load_configs().get(model_type)
    if ret is None:
        raise ValueError(f"Cannot find load config for model type {model_type}")
    return ret


def tweak_model_args(model_args):
    "Slightly modify model args for better user experience"
    model_args = {k: v for k, v in model_args.items()}  # make a copy
    if "dtype" in model_args and "param_dtype" not in model_args:
        model_args["param_dtype"] = model_args["dtype"]
    return model_args


def from_pretrained(model_id: str, revision: str | None = None, **model_args):
    """
    Load a model from a Huggingface Hub.

    Args:
        repo_id (str): ID of a model repo in Hugginface Hub.
        **model_args: Keyword arguments to overwrite defaults in ModArgsC().

    Returns:
        Tuple of (tokenizer, model, hf_config).
    """
    model_dir = snapshot_download(model_id, revision=revision, repo_type="model")
    # load config
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file) as fp:
        hf_config = json.load(fp)
    # load tokenizer
    tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_file)
    # load model
    model_type = hf_config["model_type"]
    cfg = get_load_config(model_type)
    model_args = tweak_model_args(model_args)
    args = cfg.model_args_class.from_file(config_file, **model_args)
    model = cfg.model_class(args)
    update_model_from_safe(model, cfg.rules, model_dir)
    # load chat template
    # note: in huggingface/transformers, chat template is loaded into tokenizer,
    # but we use huggingface/tokenizers, which are slightly different.
    # thus, we load the template seprately and put to hf_config
    tokenizer_config_file = os.path.join(model_dir, "tokenizer_config.json")
    with open(tokenizer_config_file) as fp:
        tok_config = json.load(fp)
        hf_config["chat_template"] = tok_config.get("chat_template", cfg.chat_template)
    return tokenizer, model, hf_config
