import logging
from datetime import datetime
import time
from dotenv import load_dotenv
import os

load_dotenv()


class AnsiColorCodes:
    INFO = "\033[36m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    TIME = "\033[34m"
    MESSAGE = "\033[95m"
    CODE = "\033[90m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add specific formatting to log messages."""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.color_codes = AnsiColorCodes()

    def formatTime(self, record, datefmt=None):
        # Convert record.created (float epoch) into a datetime object
        dt = datetime.fromtimestamp(record.created)
        
        if datefmt:
            # Use standard strftime formatting (supports %f for microseconds)
            return dt.strftime(datefmt)
        else:
            # Default fallback format with full precision
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    def format(self, record):
        msg = record.msg
        levelname = record.levelname
        if levelname == "INFO":
            color = self.color_codes.INFO
        elif levelname == "WARNING":
            color = self.color_codes.WARNING
        elif levelname == "ERROR":
            color = self.color_codes.ERROR
        else:
            color = self.color_codes.RESET

        # Formatting for timestamp
        timestamp = self.formatTime(record)
        time_section = self.color_codes.TIME + f"[{timestamp}]" + self.color_codes.RESET
        level_section = color + f"{levelname}" + self.color_codes.RESET
        message_section = self.color_codes.MESSAGE + f"{msg}" + self.color_codes.RESET
        code_section = self.color_codes.CODE + f"({record.pathname}:{record.lineno}:{record.funcName}())" + self.color_codes.RESET

        return time_section + " " + level_section + " - " + message_section + " " + code_section
    
def setup_logging():
    """Sets up the logging configuration."""
    logger = logging.getLogger()
    if os.getenv("LOGGING_LEVEL", "").upper() == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    if os.getenv("LOGGING_LEVEL", "").upper() == "DEBUG":
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    formatter = ColoredFormatter()
    # Tell the formatter to use a high-precision clock
    formatter.converter = lambda ts: time.localtime(ts)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

# Setup  logging
setup_logging()

# Load logger
logger = logging.getLogger(__name__)