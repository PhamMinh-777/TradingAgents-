"""
Cố định seed ngẫu nhiên nhằm đảm bảo tính tái lập (reproducibility).
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho random, numpy và môi trường python.

    Args:
        seed: Giá trị seed nguyên.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
