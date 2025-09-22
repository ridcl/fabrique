from fabrique.loading import LoadConfig
from fabrique.models.llama.load_rules import CHAT_TEMPLATE, RULES
from fabrique.models.llama.modeling import ModelArgs, Transformer

LOAD_CONFIG = LoadConfig(
    model_types=["llama"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES,
    chat_template=CHAT_TEMPLATE,
)
