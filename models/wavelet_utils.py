SUPPORTED_WAVELETS = ("db4", "coif2", "bior4.4", "sym8")
DEFAULT_WAVELET = "db4"
DEFAULT_WAVELET_LEVELS = 3
DEFAULT_WAVELET_THETA_INIT = 0.02


def validate_wavelet_name(name: str) -> str:
    if name not in SUPPORTED_WAVELETS:
        raise ValueError(
            f"Unsupported wavelet '{name}'. "
            f"Supported: {list(SUPPORTED_WAVELETS)}"
        )
    return name
