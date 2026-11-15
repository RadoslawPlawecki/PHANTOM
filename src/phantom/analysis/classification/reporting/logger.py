"""
The experiment log: one logger per classification run, writing to the
console and to a log file in the run's output folder; ``log_tqdm`` prints
without breaking an active progress bar.
"""

import logging
import os

from tqdm import tqdm


class Logger:
    @staticmethod
    def setup_logger(name="tool", log_dir="logs", log_filename="execution.log"):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        Logger.close(logger)  # the previous run's file handler: close, not just drop
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.info(f"Log File Initialized: {log_path}")
        return logger

    @staticmethod
    def close(logger):
        """Closes and removes every handler (their log files are complete)."""
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    @staticmethod
    def log_tqdm(logger, message):
        tqdm.write(message)
        logger.info(message)
