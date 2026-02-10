import os

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
# TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

SHOPAGENT_DEBUG = bool(os.getenv("SHOPAGENT_DEBUG", "0") == "1")