import getpass
import os
from pathlib import Path
from typing import Optional

__all__ = [
    "PathConfig",
    "get_path_config",
    "LIB_DIR",
    "CACHE_DIR",
    "OUTPUT_DIR",
    "DATA_DIR",
    "CHECKPOINT_DIR",
    "CAMPAIGN_DIR",
]


class PathConfig:
    """
    Portable path configuration that works across HPC, local, and cloud systems.

    Resolution order for each path:
    1. Environment variable (if set)
    2. HPC-specific path (if exists)
    3. Cross-platform default (always works)

    Environment variables:
        GSO_LIB_DIR        - Library/dependency storage
        GSO_CACHE_DIR      - Coefficient cache
        GSO_OUTPUT_DIR     - Simulation outputs
        GSO_DATA_DIR       - Simulation data
        GSO_CHECKPOINT_DIR - Optimization checkpoints
        GSO_CAMPAIGN_DIR   - Campaign results
        GSO_BASE_DIR       - Override base directory for all paths

    Usage:
        paths = PathConfig()
        print(paths.cache_dir)  # Returns appropriate path for current system

        # Or with custom base:
        paths = PathConfig(base_dir="/custom/path")
    """

    def __init__(self, base_dir: Optional[str] = None, verbose: bool = True):
        self._verbose = verbose
        self._username = getpass.getuser()

        # Determine base directory
        self._base_dir = self._resolve_base_dir(base_dir)

        # Initialize all paths
        self._lib_dir: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._data_dir: Optional[Path] = None
        self._checkpoint_dir: Optional[Path] = None
        self._campaign_dir: Optional[Path] = None

        # Lazily initialized
        self._initialized = False

    def _resolve_base_dir(self, override: Optional[str]) -> Path:
        """Determine the base directory for all paths."""
        # 1. Explicit override
        if override is not None:
            return Path(override)

        # 2. Environment variable
        env_base = os.environ.get("GSO_BASE_DIR")
        if env_base:
            return Path(env_base)

        # 3. HPC scratch directory (common patterns)
        hpc_scratch_patterns = [
            f"/scratch/{self._username}",  # SLURM standard
            f"/scratch/users/{self._username}",  # Some clusters
            f"/work/{self._username}",  # PBS/Torque
            f"/gpfs/scratch/{self._username}",  # GPFS-based
            f"/lustre/scratch/{self._username}",  # Lustre-based
            os.environ.get("SCRATCH", ""),  # $SCRATCH env var
            os.environ.get("WORK", ""),  # $WORK env var
        ]

        for pattern in hpc_scratch_patterns:
            if pattern and Path(pattern).exists():
                if self._verbose:
                    print(f"Detected HPC scratch directory: {pattern}")
                return Path(pattern)

        # 4. Cross-platform default: user's home directory
        home = Path.home()
        default_base = home / ".wings"

        if self._verbose:
            print(f"Using default base directory: {default_base}")

        return default_base

    def _get_path(
        self, env_var: str, subdir: str, hpc_alternatives: Optional[list[str]] = None
    ) -> Path:
        """
        Resolve a path with environment override and HPC detection.

        Args:
            env_var: Environment variable name to check
            subdir: Subdirectory name under base_dir
            hpc_alternatives: Alternative HPC paths to check
        """
        # 1. Environment variable override
        env_value = os.environ.get(env_var)
        if env_value:
            return Path(env_value)

        # 2. Check HPC alternatives
        if hpc_alternatives:
            for alt in hpc_alternatives:
                alt_path = Path(alt.format(username=self._username))
                if alt_path.parent.exists():
                    return alt_path

        # 3. Default: subdirectory of base
        return self._base_dir / subdir

    def _ensure_initialized(self) -> None:
        """Lazily initialize and create all directories."""
        if self._initialized:
            return

        # Resolve all paths
        self._lib_dir = self._get_path("GSO_LIB_DIR", "lib", ["/home/{username}/lib"])

        self._cache_dir = self._get_path(
            "GSO_CACHE_DIR", "cache/coefficients", ["/scratch/{username}/coefficient_cache"]
        )

        self._output_dir = self._get_path(
            "GSO_OUTPUT_DIR", "output", ["/scratch/{username}/simulation_output"]
        )

        self._data_dir = self._get_path(
            "GSO_DATA_DIR", "data", ["/scratch/{username}/simulation_data"]
        )

        self._checkpoint_dir = self._get_path(
            "GSO_CHECKPOINT_DIR", "checkpoints", ["/scratch/{username}/optimization_checkpoints"]
        )

        self._campaign_dir = self._get_path(
            "GSO_CAMPAIGN_DIR", "campaigns", ["/scratch/{username}/optimization_campaigns"]
        )

        # Create all directories
        all_dirs = [
            self._lib_dir,
            self._cache_dir,
            self._output_dir,
            self._data_dir,
            self._checkpoint_dir,
            self._campaign_dir,
        ]

        for directory in all_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                if self._verbose:
                    print(f"Created/verified directory: {directory}")
            except PermissionError as e:
                print(f"Warning: Cannot create {directory}: {e}")
                print(f"  Set {self._env_var_for_path(directory)} to override")

        self._initialized = True

    def _env_var_for_path(self, path: Path) -> str:
        """Get the environment variable name for a path (for error messages)."""
        mapping = {
            self._lib_dir: "GSO_LIB_DIR",
            self._cache_dir: "GSO_CACHE_DIR",
            self._output_dir: "GSO_OUTPUT_DIR",
            self._data_dir: "GSO_DATA_DIR",
            self._checkpoint_dir: "GSO_CHECKPOINT_DIR",
            self._campaign_dir: "GSO_CAMPAIGN_DIR",
        }
        return mapping.get(path, "GSO_BASE_DIR")

    @property
    def lib_dir(self) -> Path:
        self._ensure_initialized()
        return self._lib_dir

    @property
    def cache_dir(self) -> Path:
        self._ensure_initialized()
        return self._cache_dir

    @property
    def output_dir(self) -> Path:
        self._ensure_initialized()
        return self._output_dir

    @property
    def data_dir(self) -> Path:
        self._ensure_initialized()
        return self._data_dir

    @property
    def checkpoint_dir(self) -> Path:
        self._ensure_initialized()
        return self._checkpoint_dir

    @property
    def campaign_dir(self) -> Path:
        self._ensure_initialized()
        return self._campaign_dir

    def summary(self) -> str:
        """Return a summary of all configured paths."""
        self._ensure_initialized()
        lines = [
            "Path Configuration:",
            f"  Base:        {self._base_dir}",
            f"  Library:     {self._lib_dir}",
            f"  Cache:       {self._cache_dir}",
            f"  Output:      {self._output_dir}",
            f"  Data:        {self._data_dir}",
            f"  Checkpoints: {self._checkpoint_dir}",
            f"  Campaigns:   {self._campaign_dir}",
        ]
        return "\n".join(lines)


# Initialize global path configuration
# This replaces the old hardcoded paths
_path_config: Optional[PathConfig] = None


def get_path_config(base_dir: Optional[str] = None, verbose: bool = True) -> PathConfig:
    """
    Get or create the global path configuration.

    Call with base_dir to override on first use:
        paths = get_path_config(base_dir="/my/custom/path")

    Subsequent calls return the same instance:
        paths = get_path_config()  # Returns existing config
    """
    global _path_config
    if _path_config is None:
        _path_config = PathConfig(base_dir=base_dir, verbose=verbose)
    return _path_config


# For backward compatibility, create module-level path variables
# These are now properties that resolve lazily
class _LazyPath:
    """Descriptor for lazy path resolution with backward compatibility."""

    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def __get__(self, obj, objtype=None) -> Path:
        return getattr(get_path_config(verbose=False), self.attr_name)


class _PathNamespace:
    """Namespace providing backward-compatible path access."""

    lib_dir = _LazyPath("lib_dir")
    cache_dir = _LazyPath("cache_dir")
    output_dir = _LazyPath("output_dir")
    data_dir = _LazyPath("data_dir")
    checkpoint_dir = _LazyPath("checkpoint_dir")
    campaign_dir = _LazyPath("campaign_dir")


_paths = _PathNamespace()

# Backward-compatible module-level constants
# These now dynamically resolve to the correct paths
LIB_DIR = property(lambda _self: get_path_config(verbose=False).lib_dir)
CACHE_DIR = property(lambda _self: get_path_config(verbose=False).cache_dir)
OUTPUT_DIR = property(lambda _self: get_path_config(verbose=False).output_dir)
DATA_DIR = property(lambda _self: get_path_config(verbose=False).data_dir)
CHECKPOINT_DIR = property(lambda _self: get_path_config(verbose=False).checkpoint_dir)
CAMPAIGN_DIR = property(lambda _self: get_path_config(verbose=False).campaign_dir)

# Actually, for true backward compatibility at module level, use this simpler approach:
# Initialize paths on module load
_pc = get_path_config(verbose=False)
LIB_DIR = _pc.lib_dir
CACHE_DIR = _pc.cache_dir
OUTPUT_DIR = _pc.output_dir
DATA_DIR = _pc.data_dir
CHECKPOINT_DIR = _pc.checkpoint_dir
CAMPAIGN_DIR = _pc.campaign_dir
