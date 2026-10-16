import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import jaccard_score, f1_score

# otsu
def segment_otsu(index_map: np.ndarray) -> np.ndarray:
    valid_pixels = index_map[~np.isnan(index_map)]
    threshold_value = threshold_otsu(valid_pixels)

    binary_mask = (index_map > threshold_value).astype(np.uint8)
    return binary_mask, threshold_value

if __name__ == "__main__":
    from indices import calculate_ndwi