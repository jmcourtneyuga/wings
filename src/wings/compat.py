import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Conditional imports for cuQuantum
try:
    import cupy as cp
    from cuquantum.bindings import custatevec as cusv

    HAS_CUSTATEVEC = True
except ImportError:
    HAS_CUSTATEVEC = False
    cp = None
    cusv = None


class CuQuantumCompat:
    """
    Compatibility layer for cuQuantum API differences across versions.

    Handles the cudaDataType and ComputeType enum access which varies
    between cuQuantum versions (pre-24.x vs 24.x+).
    """

    # Standard CUDA data type values (from cuda_runtime_api.h)
    _CUDA_DTYPE_MAP = {
        "CUDA_R_32F": 0,
        "CUDA_R_64F": 1,
        "CUDA_C_32F": 4,
        "CUDA_C_64F": 5,
        "CUDA_C_128F": 5,  # Alias for complex128
    }

    _COMPUTE_TYPE_MAP = {
        "COMPUTE_32F": 4,
        "COMPUTE_64F": 16,
    }

    def __init__(self):
        self._cuda_c_64f: Optional[int] = None
        self._compute_64f: Optional[int] = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization to detect available API."""
        if self._initialized:
            return

        self._cuda_c_64f = self._detect_cuda_dtype()
        self._compute_64f = self._detect_compute_type()
        self._initialized = True

    def _detect_cuda_dtype(self) -> int:
        """Detect CUDA_C_64F value from cusv or fall back to known constant."""
        if not HAS_CUSTATEVEC:
            return self._CUDA_DTYPE_MAP["CUDA_C_64F"]

        # Try various API locations (differs by cuQuantum version)
        access_paths = [
            lambda: int(cusv.cudaDataType.CUDA_C_64F),
            lambda: int(cusv.cudaDataType.CUDA_C_128F),
            lambda: int(cusv.CUDA_C_64F),
            lambda: cusv.cudaDataType["CUDA_C_64F"],
        ]

        for accessor in access_paths:
            try:
                value = accessor()
                logger.debug(f"Found CUDA_C_64F via {accessor.__code__.co_consts}: {value}")
                return value
            except (AttributeError, TypeError, KeyError):
                continue

        logger.debug("Using fallback CUDA_C_64F value: 5")
        return self._CUDA_DTYPE_MAP["CUDA_C_64F"]

    def _detect_compute_type(self) -> int:
        """Detect COMPUTE_64F value from cusv or fall back to known constant."""
        if not HAS_CUSTATEVEC:
            return self._COMPUTE_TYPE_MAP["COMPUTE_64F"]

        access_paths = [
            lambda: int(cusv.ComputeType.COMPUTE_64F),
            lambda: int(cusv.COMPUTE_64F),
            lambda: cusv.ComputeType["COMPUTE_64F"],
        ]

        for accessor in access_paths:
            try:
                value = accessor()
                logger.debug(f"Found COMPUTE_64F: {value}")
                return value
            except (AttributeError, TypeError, KeyError):
                continue

        logger.debug("Using fallback COMPUTE_64F value: 16")
        return self._COMPUTE_TYPE_MAP["COMPUTE_64F"]

    @property
    def CUDA_C_64F(self) -> int:
        self._initialize()
        return self._cuda_c_64f

    @property
    def COMPUTE_64F(self) -> int:
        self._initialize()
        return self._compute_64f


# Global compatibility instance
cuquantum_compat = CuQuantumCompat()


# Convenience accessors (replaces the old global variables)
def get_cuda_dtype() -> int:
    """Get the CUDA data type constant for complex128."""
    return cuquantum_compat.CUDA_C_64F


def get_compute_type() -> int:
    """Get the compute type constant for 64-bit precision."""
    return cuquantum_compat.COMPUTE_64F


__all__ = [
    "HAS_CUSTATEVEC",
    "CuQuantumCompat",
    "cuquantum_compat",
    "get_cuda_dtype",
    "get_compute_type",
]
