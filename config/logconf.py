import logging
import os
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler

from config.constants import LOG_DIR
from utils.display import format_duration

# Color mapping for console output
LOG_COLORS = {
    "DEBUG": "\033[92m",  # Green
    "INFO": "\033[94m",  # Blue
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
    "ELAPSED": "\033[96m",  # Cyan (right-aligned clock)
    "ENDC": "\033[0m",  # Reset
}

class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)
    def flush(self):
        pass

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages and elapsed time.
    This formatter uses ANSI escape codes to colorize the log messages based on their severity level.
    It also includes a right-aligned clock that shows the elapsed time since the logger was initialized.
    The elapsed time is displayed in a human-readable format (e.g., "1h 23m 45s").
    The formatter is designed to be used with a logger that has a console handler.
    The elapsed time is calculated from the time the logger was initialized and is displayed in a right-aligned format.
    The formatter also ensures that the log messages are padded to a specified width, which can be adjusted using the `width` parameter.
    The `remove_ansi` method is used to strip ANSI escape codes from the log message for accurate padding calculation.
    The `format` method is overridden to customize the log message format, including the timestamp, logger name, log level, and message.
    The `setup_logger` function is used to configure the logger with a file handler and a stream handler.
    The file handler writes log messages to a specified log file, while the stream handler outputs log messages to the console.
    The logger is set to the specified logging level, and the log file is created in the specified directory.
    The log file is rotated based on size, and old log files are backed up.
    """

    def __init__(self, fmt=None, datefmt=None, width=200):
        super().__init__(fmt, datefmt)
        self.start_time = datetime.now()
        self.width = width

    def format(self, record):
        """
        Format the log record with colors and elapsed time.
        This method overrides the default format method to customize the log message format.
        It includes the timestamp, logger name, log level, and message.
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = f"{LOG_COLORS['ELAPSED']}⏱ {format_duration(elapsed)}{LOG_COLORS['ENDC']}"

        # Compose colored parts
        color = LOG_COLORS.get(record.levelname, LOG_COLORS["INFO"])
        time_str = f"{LOG_COLORS['DEBUG']}{self.formatTime(record)}{LOG_COLORS['ENDC']}"
        name_str = f"{LOG_COLORS['WARNING']}{record.name}{LOG_COLORS['ENDC']}"
        level_str = f"{color}{record.levelname}{LOG_COLORS['ENDC']}"
        msg_str = f"{color}{record.getMessage()}{LOG_COLORS['ENDC']}"

        raw_msg = f"{time_str} - {name_str} - {level_str} - {msg_str}"
        no_ansi_len = len(self.remove_ansi(raw_msg))
        padding = max(0, self.width - no_ansi_len)
        return f"{raw_msg}{' ' * padding}{elapsed_str}"

    @staticmethod
    def remove_ansi(s):
        """
        Remove ANSI escape codes from a string.
        """
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', s)


def setup_logger(
        name="phoskintime",
        log_file=None,
        level=logging.DEBUG,
        log_dir=LOG_DIR,
        rotate=True,
        max_bytes=2 * 1024 * 1024,
        backup_count=5,
        mp_file_logging="main_only", # off | main_only | per_process
):
    """
    Setup a logger with colored output and file logging.
    This function creates a logger with colored output for console messages
    :param name:
    :param log_file:
    :param level:
    :param log_dir:
    :param rotate:
    :param max_bytes:
    :param backup_count:
    :param mp_file_logging:
        - "off": disable file logging
        - "main_only": file logging only in main process
        - "per_process": file logging in each process
    :return: logger
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # IMPORTANT: do not nuke handlers that might be inherited incorrectly unless you re-add safely
    if logger.hasHandlers():
        logger.handlers.clear()

    # Detect multiprocessing worker
    # (safe even if you don't use multiprocessing in some runs)
    is_worker = False
    try:
        import multiprocessing as mp
        is_worker = (mp.current_process().name != "MainProcess")
    except Exception:
        is_worker = False

    # Decide file logging policy
    enable_file = True
    if mp_file_logging == "off":
        enable_file = False
    elif mp_file_logging == "main_only" and is_worker:
        enable_file = False
    elif mp_file_logging == "per_process" and is_worker:
        enable_file = True

    # File Handler (ONLY when enabled)
    if enable_file:
        if mp_file_logging == "per_process" and is_worker:
            pid = os.getpid()
            base, ext = os.path.splitext(log_file)
            log_file_use = f"{base}.pid{pid}{ext}"
        else:
            log_file_use = log_file

        # Avoid RotatingFileHandler across processes on NFS.
        # Keep rotation ONLY in the main process.
        if rotate and (not is_worker):
            file_handler = RotatingFileHandler(
                log_file_use, maxBytes=max_bytes, backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file_use)

        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Stream Handler (Console) — safe in all processes
    stream_handler = logging.StreamHandler()
    stream_format = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Prevent double logging via root handlers
    logger.propagate = False

    return logger
