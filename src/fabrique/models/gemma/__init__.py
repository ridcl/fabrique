from fabrique.loading import LoadConfig
from fabrique.models.gemma.loader import load_gemma


LOAD_CONFIG = LoadConfig(
    model_re="gemma.*",
    loader=load_gemma,
)