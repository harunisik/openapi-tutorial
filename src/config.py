import os

OPENAI_MODEL = os.getenv("LANGUAGE_MODEL_NAME", "gpt-4.1-mini")
# TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

SHOPAGENT_DEBUG = bool(os.getenv("SHOPAGENT_DEBUG", "0") == "1")