import os
import logging
from logging.handlers import RotatingFileHandler
from config.logconf import setup_logger, ColoredFormatter

def logger_creation_defaults(tmp_path):
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="unittest_logger", log_dir=log_directory)
    assert logger.name == "unittest_logger"
    file_handler = None
    stream_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handler = handler
        elif isinstance(handler, logging.FileHandler):
            file_handler = handler
    assert file_handler is not None
    assert stream_handler is not None
    assert os.path.exists(log_directory)

def logger_without_rotation_creates_regular_file_handler(tmp_path):
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="unittest_logger_no_rotate", log_dir=log_directory, rotate=False)
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
    assert file_handler is not None
    assert not isinstance(file_handler, RotatingFileHandler)

def colored_formatter_provides_ansi_output():
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    record = logging.LogRecord("color_test", logging.INFO, "", 0, "colored message", None, None)
    formatted = formatter.format(record)
    assert "\033" in formatted

def logger_emits_log_message_to_stream(capsys, tmp_path):
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="unittest_logger_stream", log_dir=log_directory)
    logger.info("stream test message")
    captured = capsys.readouterr().out
    assert "stream test message" in captured