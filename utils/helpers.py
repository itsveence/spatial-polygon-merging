import time
from datetime import datetime, timedelta
from functools import wraps
import tracemalloc
from utils.logger import logger


def time_it(func):
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        time_elapsed = (datetime.min + timedelta(seconds=elapsed)).strftime("%H:%M:%S.%f")
        logger.info(
            f"\033[32mExecution time for {func.__name__}(): {time_elapsed}\033[0m"
        )
        return result

    return wrapper

def size_it(func):
    """Decorator to measure the size of the output of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        _, peak_memory = tracemalloc.get_traced_memory()
        logger.info(
            f"\033[32mPeak memory usage for {func.__name__}(): {peak_memory / 1024 / 1024:.2f} MB\033[0m"
        )
        return result

    return wrapper