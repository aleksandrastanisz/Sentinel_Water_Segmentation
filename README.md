# Sentinel-2 Surface Water Segmentation: U-Net vs Otsu Baseline

Segmentation of surface water in Sentinel-2 Level-2A imagery with a U-Net (ResNet-34 encoder) trained on 7 input channels, benchmarked against a classical Otsu threshold on NDWI. The model is evaluated on water bodies it never saw during training, including a mountain reservoir where the reference labels themselves turn out to be unreliable.

## Data

Imagery is queried from the Microsoft Planetary Computer STAC API (`sentinel-2-l2a`, summer 2023, least cloudy scene per area, all used scenes have 0–0.1% cloud cover). Each area is cut into non-overlapping 256 x 256 px tiles at 10 m resolution.

Every area goes entirely into one split, so the same water body never appears in both training and evaluation:

| Split | Area | Tiles |
|---|---|---|
| train | Szczecin Lagoon | 21 |
| train | Lake Śniardwy | 27 |
| train | Vistula Lagoon | 14 |
| train | Włocławek Reservoir | 11 |
| val | Lake Gopło | 12 |
| test | Jeziorsko Reservoir | 10 |
| test | Solina Reservoir (mountain valley) | 8 |

Tiles are kept only if they contain 2–98% water and less than 5% invalid pixels (no data, clouds, cloud shadows, cirrus).

**Labels** come from the Sen2Cor Scene Classification Layer (SCL class 6 = water), resampled from 20 m to 10 m with nearest neighbour. SCL is available for every scene for free, but it is a classification product, not verified ground truth (see Discussion).

**Input channels (7):**
- B02, B03, B04, B08 (10 m)
- B11 (SWIR, 20 m, bilinearly resampled to 10 m)
- `NDWI = (B03 - B08) / (B03 + B08)`
- `MNDWI = (B03 - B11) / (B03 + B11)`

Reflectance is corrected for the processing baseline 04.00+ offset (DN - 1000, read from scene metadata) and scaled to the range 0–1.

## Training

- U-Net with ResNet-34 encoder (ImageNet weights), via `segmentation-models-pytorch`
- Loss: 0.5 × BCE + 0.5 × Dice
- AdamW, learning rate 1e-4, weight decay 1e-4, batch size 4, 25 epochs
- Augmentations: random horizontal/vertical flips and 90° rotations (geometric only, spectral values unchanged)
- Fixed random seeds; checkpoint selected by validation IoU
- Trained on CPU

## Results (test set)

Metrics are computed globally over all pixels of each area. The baseline thresholds NDWI with Otsu's method separately for each tile.

| Area | Method | IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|
| Jeziorsko | Otsu (NDWI) | 0.951 | 0.975 | 0.960 | 0.990 |
| Jeziorsko | U-Net | 0.949 | 0.974 | 0.954 | 0.994 |
| Solina | Otsu (NDWI) | 0.833 | 0.909 | 0.833 | 1.000 |
| Solina | U-Net | **0.886** | **0.939** | **0.894** | 0.989 |
| **All test tiles** | Otsu (NDWI) | 0.909 | 0.953 | 0.915 | 0.993 |
| **All test tiles** | U-Net | **0.928** | **0.963** | **0.935** | 0.992 |

![Test set examples](./outputs/figures/evaluation_test.png)

Full per-area metrics: [`outputs/tables/metrics_test.csv`](./outputs/tables/metrics_test.csv)

## Discussion

**Lowland reservoir (Jeziorsko):** both methods perform almost identically. On a clear summer scene with an open water surface, a simple NDWI threshold is already a strong baseline.

**Mountain reservoir (Solina):** the SCL reference misses large parts of the reservoir. Water in the shaded valley is clearly visible in the RGB image, but SCL marks only fragments of it as water. Both methods segment the full water surface correctly, so their lower precision here mostly reflects errors in the reference, not in the predictions. U-Net scores higher mainly because it produces fewer small false-positive patches on land than Otsu.

This shows the main weakness of using SCL as ground truth: in difficult conditions (terrain shadows) the metrics measure agreement with Sen2Cor rather than correctness.

## Limitations

- **Reference labels:** SCL is a classifier output, not verified ground truth. It is used for both training and evaluation, so the model partly learns to reproduce SCL behaviour, including its errors. An independent reference (e.g. ESA WorldCover or manual annotation) would make the test results more reliable.
- **Small test set:** two reservoirs (18 tiles). Results show a trend, not a general performance estimate.
- **Single season:** all scenes are from summer 2023 with almost no clouds. Winter, ice, turbid water and cloudy conditions are not covered.
- **Tile filter:** tiles with less than 2% or more than 98% water are excluded, which removes both the hardest and the most trivial cases.
- **Checkpoint selection:** the best model is chosen by IoU on a single validation lake (12 tiles), which is a noisy criterion.

## Project structure

```
src/
  download_data.py   # STAC query, band reading, offset correction, tiling into train/val/test
  indices.py         # NDWI, MNDWI and 7-channel stacking
  dataset.py         # PyTorch Dataset
  model.py           # U-Net definition
  train.py           # training loop with augmentations
  evaluate.py        # U-Net vs Otsu on the test set, per-area metrics and figure
```

## Setup & Run

Tested with Python 3.13.

```bash
pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/evaluate.py
```

## Data source

Contains modified Copernicus Sentinel-2 data (2023), processed by ESA, accessed via Microsoft Planetary Computer.
