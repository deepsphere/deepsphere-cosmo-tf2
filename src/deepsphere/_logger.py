import logging
import os
import sys

# the logger
logger = logging.getLogger("sbi_flows")
log_formatter = logging.Formatter(
    fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ", datefmt="%y-%m-%d %H:%M:%S", style="%"
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False
logger.setLevel(logging.INFO)

# set the level from environment variable if set
if "DEEPSPHERE_LOG_LEVEL" in os.environ:
    log_level = os.environ["DEEPSPHERE_LOG_LEVEL"]
    try:
        # try to get an int
        log_level = int(log_level)
    except ValueError:
        logger.warning(f"Loglevel set in DEEPSPHERE_LOG_LEVEL is not an int, got {log_level}. Using default INFO!")
        log_level = 4

    # set the level
    if log_level <= 1:
        log_level = logging.CRITICAL
    elif log_level == 2:
        log_level = logging.ERROR
    elif log_level == 3:
        log_level = logging.WARNING
    elif log_level == 4:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger.setLevel(log_level)
