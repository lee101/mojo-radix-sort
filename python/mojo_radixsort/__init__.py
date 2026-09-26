"""Mojo-backed stable radix sort for NumPy arrays.

Importable alongside any other package; nothing here shadows `numpy` or
`radix_sort`. See README.md for what is and is not covered.
"""

from .core import (
    DEFAULT_BITS,
    argsort,
    bytes_histogram,
    digit_histogram,
    key_width,
    order_keys,
    sort,
    sort_pairs,
)
from ._lib import lib

__version__ = "0.1.0"
__all__ = [
    "argsort",
    "sort",
    "sort_pairs",
    "order_keys",
    "digit_histogram",
    "bytes_histogram",
    "key_width",
    "DEFAULT_BITS",
    "lib",
    "__version__",
]
