import logging
import os


def setup_logger(name: str, log_file: str, level=logging.INFO, verbose: bool = False) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    if verbose:
        # Console handler for INFO level
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger
