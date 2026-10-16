import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import jaccard_score, f1_score

# otsu
def segment_otsu(index_map: np.ndarray) -> np.ndarray:
    valid_pixels = index_map[~np.isnan(index_map)]
    threshold_value = threshold_otsu(valid_pixels)

    binary_mask = (index_map > threshold_value).astype(np.uint8)
    return binary_mask, threshold_value

# oblicza iou i dice
def evaluate_baseline(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict:
    pred_flat = pred_mask.flatten()
    target_flat = target_mask.flatten()

    iou = jaccard_score(target_flat, pred_flat, average="binary", zero_division=0)
    dice = f1_score(target_flat, pred_flat, average="binary", zero_division=0)

    return {
        "IoU": float(iou),
        "Dice": float(dice)
    }


if __name__ == "__main__":
    from indices import calculate_ndwi