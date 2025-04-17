import os
import logging
from config.logconf import setup_logger, ColoredFormatter

def logger_handles_empty_log_directory(tmp_path):
    """
    Test that the logger creates the log directory if it does not exist.
    """
    log_directory = str(tmp_path / "nonexistent_logs")
    logger = setup_logger(name="empty_log_dir_test", log_dir=log_directory)
    assert os.path.exists(log_directory)


def logger_does_not_duplicate_handlers(tmp_path):
    """
    Test that the logger does not add duplicate handlers when called multiple times.
    """
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="duplicate_handler_test", log_dir=log_directory)
    initial_handler_count = len(logger.handlers)
    logger = setup_logger(name="duplicate_handler_test", log_dir=log_directory)
    assert len(logger.handlers) == initial_handler_count


def logger_handles_no_log_directory():
    """
    Test that the logger works without a log directory and only uses a stream handler.
    """
    logger = setup_logger(name="no_log_dir_test", log_dir=None)
    file_handler = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    assert not file_handler


def logger_respects_custom_formatter(tmp_path):
    """
    Test that the logger uses the custom formatter for all handlers.
    """
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="custom_formatter_test", log_dir=log_directory)
    for handler in logger.handlers:
        assert isinstance(handler.formatter, ColoredFormatter)