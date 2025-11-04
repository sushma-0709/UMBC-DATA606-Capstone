"""
Snowflake connection loader and data pipeline.

This module establishes connections to Snowflake, validates credentials,
and provides utilities for data loading operations.

Usage:
    python Scripts/load_data.py
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector import errors as snowflake_errors

# Configure logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure application-wide logging."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "load_data.log",
        maxBytes=10_000_000,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def load_snowflake_config() -> Dict[str, str]:
    """
    Load and validate Snowflake configuration from environment variables.

    Returns:
        Dictionary with Snowflake connection parameters

    Raises:
        EnvironmentError: If required environment variables are missing
    """
    load_dotenv()

    required_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)

    config = {var: os.getenv(var) for var in required_vars}
    logger.info("Snowflake configuration loaded successfully")

    return config


def test_snowflake_connection(config: Dict[str, str]) -> bool:
    """
    Test Snowflake connection with proper error handling.

    Args:
        config: Snowflake connection configuration dictionary

    Returns:
        True if connection successful, False otherwise
    """
    conn = None
    cur = None

    try:
        logger.info("Attempting to connect to Snowflake...")

        conn = snowflake.connector.connect(
            user=config["SNOWFLAKE_USER"],
            password=config["SNOWFLAKE_PASSWORD"],
            account=config["SNOWFLAKE_ACCOUNT"],
            warehouse=config["SNOWFLAKE_WAREHOUSE"],
            database=config["SNOWFLAKE_DATABASE"],
            schema=config["SNOWFLAKE_SCHEMA"],
        )

        cur = conn.cursor()
        logger.info("Connected successfully")

        # Test queries
        test_queries = {
            "Version": "SELECT CURRENT_VERSION()",
            "User": "SELECT CURRENT_USER()",
            "Account": "SELECT CURRENT_ACCOUNT()",
            "Database": "SELECT CURRENT_DATABASE()",
            "Schema": "SELECT CURRENT_SCHEMA()",
            "Warehouse": "SELECT CURRENT_WAREHOUSE()",
        }

        logger.info("Running test queries...")

        for key, query in test_queries.items():
            try:
                cur.execute(query)
                result = cur.fetchone()

                if result and len(result) > 0:
                    logger.info(f"{key}: {result[0]}")
                else:
                    logger.warning(f"{key}: No result returned")

            except snowflake_errors.ProgrammingError as e:
                logger.error(f"Query failed for {key}: {e}")
                return False

        logger.info("✅ Snowflake connection test PASSED")
        return True

    except snowflake_errors.DatabaseError as e:
        logger.error(f"Database connection error: {e}")
        return False

    except snowflake_errors.OperationalError as e:
        logger.error(f"Operational error (check credentials/account): {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error during connection test: {e}", exc_info=True)
        return False

    finally:
        # Cleanup
        if cur:
            try:
                cur.close()
            except Exception as e:
                logger.warning(f"Error closing cursor: {e}")

        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def main() -> int:
    """
    Main entry point for testing Snowflake connection.

    Returns:
        0 if successful, 1 if failed
    """
    try:
        config = load_snowflake_config()

        if test_snowflake_connection(config):
            logger.info("All tests passed")
            return 0
        else:
            logger.error("Connection test failed")
            return 1

    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
