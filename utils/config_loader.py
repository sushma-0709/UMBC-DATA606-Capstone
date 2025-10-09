import os
import toml
from dotenv import load_dotenv

def load_snowflake_config():
    # Load env vars from .env
    load_dotenv()

    # Load TOML
    raw_config = toml.load("snowflakecli.toml")

    # Expand env vars
    expanded = {}
    for conn_name, settings in raw_config["connections"].items():
        expanded[conn_name] = {k: os.path.expandvars(v) for k, v in settings.items()}

    return expanded
