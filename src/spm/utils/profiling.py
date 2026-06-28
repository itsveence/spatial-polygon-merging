import time
from datetime import datetime, timedelta
from functools import wraps
from memory_profiler import memory_usage
from spm.utils.logger import logger


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
        baseline = memory_usage(max_usage=True)
        peak, result = memory_usage((func, args, kwargs), max_usage=True, retval=True)
        peak_memory = peak - baseline
        logger.info(
            f"\033[32mPeak memory usage for {func.__name__}(): {peak_memory:.2f} MB\033[0m"
        )
        return result

    return wrapper