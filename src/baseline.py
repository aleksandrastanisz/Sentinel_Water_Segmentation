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

    # test
    h, w = 128, 128
    dummy_green = np.random.uniform(0.1, 0.4, (h, w)).astype(np.float32)
    dummy_nir = np.random.uniform(0.1, 0.4, (h, w)).astype(np.float32)

    # symulacja rzeki lub jeziora
    dummy_nir[40:70, :] = 0.02
    dummy_green[40:70, :] = 0.25

    dummy_target = np.zeros((h, w), dtype=np.uint8)
    dummy_target[40:70, :] = 1

    ndwi = calculate_ndwi(dummy_green, dummy_nir)
    pred, thresh = segment_otsu(ndwi)
    metrics = evaluate_baseline(pred, dummy_target)

    print(f"Wyliczony prog Otsu: {thresh:.4f}")
    print(f"Wyniki: IoU = {metrics['IoU']:.4f}, Dice = {metrics['Dice']:.4f}")