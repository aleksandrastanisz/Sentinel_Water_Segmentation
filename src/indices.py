import numpy as np

def calculate_ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)

    numerator = green - nir
    denominator = green + nir + eps

    ndwi = numerator / denominator
    return np.clip(ndwi, -1.0, 1.0)

def calculate_mndwi(green: np.ndarray, swir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    green = green.astype(np.float32)
    swir = swir.astype(np.float32)

    numerator = green - swir
    denominator = green + swir + eps

    mndwi = numerator / denominator
    return np.clip(mndwi, -1.0, 1.0)

def stack_multispectral_tensor(
        b02: np.ndarray,
        b03: np.ndarray,
        b04: np.ndarray,
        b08: np.ndarray,
        b11: np.ndarray
) -> np.ndarray:
    ndwi = calculate_ndwi(b03, b08)
    mndwi = calculate_mndwi(b03, b11)

    tensor = np.stack(
        [
            b02.astype(np.float32),
            b03.astype(np.float32),
            b04.astype(np.float32),
            b08.astype(np.float32),
            b11.astype(np.float32),
            ndwi,
            mndwi,
        ],
        axis=-1
    )
    return tensor