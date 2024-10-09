from functools import wraps
from time import time


def timing(f):
    """Time function execution.

    Parameters
    ----------
    f : _type_

    Returns
    -------
    _type_

    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func: {f.__name__} took: {te-ts:2.4f}2.4f sec")
        return result

    return wrap
