"""
Times how long calculations take
"""

import time
from contextlib import contextmanager

@contextmanager
def timer():
    """
    Timer context manager.

    How to use in code:

    with timer() as t:
        code()

    print(t["elapsed"])
    """

    result = {}
    start = time.perf_counter()
    yield result
    end = time.perf_counter()

    result["elapsed"] = end - start