import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
ParameterArray = NDArray[np.float64]

__all__ = [
    "ComplexArray",
    "FloatArray",
    "ParameterArray",
    "NDArray",
]
