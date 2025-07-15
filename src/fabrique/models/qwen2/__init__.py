from fabrique.loading import LoadConfig
from fabrique.models.qwen2.load_rules import CHAT_TEMPLATE, RULES
from fabrique.models.qwen2.modeling import ModelArgs, Transformer

LOAD_CONFIG = LoadConfig(
    model_types=["qwen2"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES,
    chat_template=CHAT_TEMPLATE,
)
