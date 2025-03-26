import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

# Color mapping for console output
LOG_COLORS = {
    "DEBUG": "\033[92m",    # Green
    "INFO": "\033[94m",     # Blue
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "CRITICAL": "\033[95m", # Magenta
    "ENDC": "\033[0m",      # Reset
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level_color = LOG_COLORS.get(record.levelname, "")
        end_color = LOG_COLORS["ENDC"]
        record.msg = f"{level_color}{record.msg}{end_color}"
        return super().format(record)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # Write using tqdm-safe print
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    name=__name__,
    log_file=None,
    level=logging.DEBUG,
    log_dir="../logs",
    rotate=True,
    max_bytes=2 * 1024 * 1024,
    backup_count=5
):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    if rotate:
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    else:
        file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # TQDM-aware stream handler for console
    tqdm_handler = TqdmLoggingHandler()
    stream_format = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tqdm_handler.setFormatter(stream_format)
    tqdm_handler.setLevel(logging.INFO)
    logger.addHandler(tqdm_handler)

    return logger