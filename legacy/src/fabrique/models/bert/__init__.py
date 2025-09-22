from fabrique.loading import LoadConfig
from fabrique.models.bert.load_rules import RULES
from fabrique.models.bert.modeling import ModelArgs, Transformer

LOAD_CONFIG = LoadConfig(
    model_types=["bert"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES,
)
