import os
import logging
from config.logconf import setup_logger, ColoredFormatter


def test_logger_handles_empty_log_directory(tmp_path):
    """
    Test that the logger creates the log directory if it does not exist.
    """
    log_directory = str(tmp_path / "nonexistent_logs")
    logger = setup_logger(name="empty_log_dir_test", log_dir=log_directory)
    assert os.path.exists(log_directory)


def test_logger_does_not_duplicate_handlers(tmp_path):
    """
    Test that the logger does not add duplicate handlers when called multiple times.
    """
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="duplicate_handler_test", log_dir=log_directory)
    initial_handler_count = len(logger.handlers)
    logger = setup_logger(name="duplicate_handler_test", log_dir=log_directory)
    assert len(logger.handlers) == initial_handler_count


def test_logger_handles_no_log_directory():
    """
    Test that the logger works without a log directory and only uses a stream handler.
    """
    logger = setup_logger(name="no_log_dir_test", log_dir=None)
    file_handler = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    assert not file_handler


def test_logger_respects_custom_formatter(tmp_path):
    """
    Test that the console handler uses the coloured formatter while the file
    handler uses a plain formatter (no ANSI escape codes written to log files).
    """
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="custom_formatter_test", log_dir=log_directory)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    console_handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert len(console_handlers) == 1
    assert isinstance(console_handlers[0].formatter, ColoredFormatter)
    assert isinstance(file_handlers[0].formatter, logging.Formatter)
    assert not isinstance(file_handlers[0].formatter, ColoredFormatter)
