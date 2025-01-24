import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def timeit(func):
    """Time function calls."""

  @wraps(func)
  def wrapper(*args, **kwargs):
    """Wrapper to do work."""
    start = time.perf_counter()
    original_return_val = func(*args,**kwargs)
    end=time.perf_counter()
    elapsed_time = f"{(end-start)}"
    logger.info("Time elapsed in %s: %s (s)", func.__name__, elapsed_time)
    return original_return_val

  return wrapper
