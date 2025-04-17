import os
import logging
from logging.handlers import RotatingFileHandler
from config.logconf import setup_logger, ColoredFormatter


def test_logger_creation_defaults(tmp_path):
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


def test_logger_without_rotation_creates_regular_file_handler(tmp_path):
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="unittest_logger_no_rotate", log_dir=log_directory, rotate=False)
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
    assert file_handler is not None
    assert not isinstance(file_handler, RotatingFileHandler)


def test_colored_formatter_provides_ansi_output():
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    record = logging.LogRecord("color_test", logging.INFO, "", 0, "colored message", None, None)
    formatted = formatter.format(record)
    assert "\033" in formatted


def test_logger_emits_log_message_to_stream(capsys, tmp_path):
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="unittest_logger_stream", log_dir=log_directory)
    logger.info("stream test message")
    captured = capsys.readouterr().out
    assert "stream test message" in captured


# Additional tests for package configuration

def test_logger_with_custom_level(capsys, tmp_path):
    """
    Test that logger only emits messages at or above the custom log level.
    """
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="custom_level", log_dir=log_directory, level=logging.WARNING)
    logger.info("this info should not appear")
    logger.warning("this warning should appear")
    captured = capsys.readouterr().out
    assert "this warning should appear" in captured
    assert "this info should not appear" not in captured


def test_file_logging(tmp_path):
    """
    Test that a message logged via the file handler is written to the file.
    """
    log_directory = str(tmp_path / "logs")
    logger = setup_logger(name="file_logger", log_dir=log_directory)
    log_message = "file log test message"
    logger.info(log_message)

    # Retrieve the file handler to get the log file path
    file_handler = next(
        handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
    )
    file_path = file_handler.baseFilename

    # Wait until the file is created and written to
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        file_content = f.read()
    assert log_message in file_content