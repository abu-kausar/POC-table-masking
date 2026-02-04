import logging
from logging.handlers import RotatingFileHandler
import os


class Logger:
    _loggers = {}

    @staticmethod
    def get_logger(
        name: str,
        level=logging.INFO,
        log_file="app.log"
    ):
        # Reuse logger if it already exists
        if name in Logger._loggers:
            return Logger._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # prevent duplicate logs

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler (rotates when file gets big)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        Logger._loggers[name] = logger
        return logger
# Example usage:
# logger = Logger.get_logger("my_app", log_file="my_app.log")
# logger.info("This is an info message.") 