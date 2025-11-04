"""Configuration loader for Snowflake connections.

This module provides utilities to load and validate Snowflake connection
configurations from environment variables and TOML files.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import toml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_snowflake_config(
    config_path: str = "snowflakecli.toml",
) -> Dict[str, Dict[str, str]]:
    """
    Load Snowflake configuration from TOML file with environment variable expansion.

    Args:
        config_path: Path to snowflakecli.toml file

    Returns:
        Dictionary of connection configurations with expanded environment variables

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required env vars are missing or TOML is invalid
        toml.TomlDecodeError: If TOML syntax is invalid
    """
    # Load environment variables from .env
    load_dotenv()

    # Validate config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Please create it or check the path."
        )

    if not config_file.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    logger.info(f"Loading configuration from: {config_file.absolute()}")

    # Load and parse TOML
    try:
        raw_config = toml.load(config_path)
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML syntax in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {config_path}: {e}")

    # Validate structure
    if "connections" not in raw_config:
        raise ValueError(
            "No 'connections' section found in configuration file. "
            "Expected: [connections.connection_name]"
        )

    # Expand environment variables and validate
    expanded = {}
    required_fields = ["account", "user", "password"]

    for conn_name, settings in raw_config["connections"].items():
        if not isinstance(settings, dict):
            logger.warning(f"Skipping invalid connection config: {conn_name}")
            continue

        try:
            # Expand environment variables in all string values
            expanded[conn_name] = {}
            for k, v in settings.items():
                if isinstance(v, str):
                    expanded_value = os.path.expandvars(v)
                    expanded[conn_name][k] = expanded_value
                else:
                    expanded[conn_name][k] = v

            # Validate required fields exist and are not empty
            missing_fields = []
            for field in required_fields:
                if field not in expanded[conn_name]:
                    missing_fields.append(field)
                elif not expanded[conn_name][field]:
                    # Empty or unresolved env var
                    missing_fields.append(field)
                elif expanded[conn_name][field].startswith("$"):
                    # Unresolved environment variable
                    missing_fields.append(f"{field} (unresolved env var)")

            if missing_fields:
                raise ValueError(
                    f"Connection '{conn_name}' missing or invalid values for: {missing_fields}. "
                    f"Check environment variables and TOML configuration."
                )

            logger.info(f"Loaded connection: {conn_name}")

        except ValueError as e:
            logger.error(f"Failed to process connection '{conn_name}': {e}")
            raise

    if not expanded:
        raise ValueError("No valid connections found in configuration file")

    logger.info(f"Successfully loaded {len(expanded)} connection configuration(s)")
    return expanded


def get_connection_config(
    config_path: str = "snowflakecli.toml",
    connection_name: str = "default",
) -> Dict[str, str]:
    """
    Get a specific connection configuration by name.

    Args:
        config_path: Path to snowflakecli.toml file
        connection_name: Name of the connection to retrieve

    Returns:
        Dictionary with connection parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If connection name not found
    """
    all_connections = load_snowflake_config(config_path)

    if connection_name not in all_connections:
        available = ", ".join(all_connections.keys())
        raise ValueError(
            f"Connection '{connection_name}' not found. "
            f"Available connections: {available}"
        )

    return all_connections[connection_name]
