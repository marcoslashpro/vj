import json
from pathlib import Path
import logging.config
import warnings

import os
from dotenv import load_dotenv


load_dotenv()


LOG_CONFIG = Path(__file__).parent / "config.json"


class ConfigError(Exception):
    pass


class ConfigNotFoundError(ConfigError):
    pass


class InvalidConfigError(ConfigError):
    pass


def setup_logging():
    if not LOG_CONFIG.exists():
        raise ConfigNotFoundError(f"Missing logging config file at {LOG_CONFIG}")

    if not (config := LOG_CONFIG.read_text()):
        raise InvalidConfigError(f"Invalid emptylog config file at {LOG_CONFIG}")

    try:
        json_config = json.loads(config)
    except json.JSONDecodeError as e:
        raise InvalidConfigError(f"Error while parsing the config file: {e}")

    level = "DEBUG" if os.environ.get("VERBOSE") else None
    if level is not None:
        if not "handlers" in json_config:
            warnings.warn(
                f"Unable to configure logging level, missing `handlers` key in config"
            )
            logging.config.dictConfig(json_config)
            return
        if not "console" in json_config["handlers"]:
            warnings.warn(
                f"Unable to configure logging level, missing `console` handler in config"
            )
            logging.config.dictConfig(json_config)
            return
        json_config["handlers"]["console"]["level"] = level
        logging.config.dictConfig(json_config)
        return
    logging.config.dictConfig(json_config)


_SETUP = False


def get_logger() -> logging.Logger:
    global _SETUP
    if not _SETUP:
        setup_logging()
        _SETUP = True

    return logging.getLogger("tj")
