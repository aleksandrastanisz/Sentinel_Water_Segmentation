\# Sentinel-2 Surface Water Segmentation \& Otsu Baseline Analysis



Deep learning pipeline segmenting surface water bodies from Sentinel-2 Level-2A multispectral imagery using a U-Net architecture with a ResNet-34 backbone\[cite: 1, 7]. The project includes a comparative study benchmarking the 7-channel deep learning model against classical adaptive Otsu thresholding on the Normalized Difference Water Index (NDWI)\[cite: 1, 2, 5].



\## Overview \& Remote Sensing Motivation

Broad-band optical and near-infrared reflectance over inland and coastal waters is susceptible to high turbidity, sediment plumes, submerged vegetation, and cloud shadows\[cite: 1, 10]. Standard index thresholding methods often produce false negatives across complex shorelines\[cite: 1, 10]. This project investigates whether a multispectral convolutional network combining native 10 m bands, resampled SWIR, and non-linear water indices can suppress radiometric noise and recover clean shoreline geometry from lower-resolution reference data\[cite: 1, 4, 7, 10].



\## Dataset \& Features

Sentinel-2 L2A scene `S2B\_MSIL2A\_20230610T101609\_R065\_T33UVV` (Szczecin Lagoon, cloud cover < 5%) queried via Planetary Computer STAC API\[cite: 1, 4]. A 2239 x 1994 px spatial crop was tiled into non-overlapping 256 x 256 px patches filtered for water presence ($0.02 \\le \\text{water\\\_ratio} \\le 0.98$) and partitioned into 15 training and 4 validation tiles\[cite: 1, 4]. Ground truth labels were derived from the Sen2Cor Scene Classification Layer (SCL class 6)\[cite: 1, 4].



Input feature vectors (7 channels):

\- Optical \& NIR bands: B02 (490 nm, 10 m), B03 (560 nm, 10 m), B04 (665 nm, 10 m), B08 (842 nm, 10 m)\[cite: 1, 4, 6]

\- Short-Wave Infrared: B11 (1610 nm, 20 m bilinearly resampled to 10 m)\[cite: 1, 4, 6]

\- Spectral indices: $\\text{NDWI} = (B03 - B08) / (B03 + B08 + 10^{-7})$ and $\\text{MNDWI} = (B03 - B11) / (B03 + B11 + 10^{-7})$\[cite: 1, 6]

\- Radiometric calibration: Processing Baseline 04.00 offset correction ($\\text{DN} - 1000$) scaled to BOA reflectance $\\in \[0.0, 1.0]$\[cite: 1, 4]



\## Results \& Comparative Evaluation

Trained over 25 epochs using compound loss ($0.5 \\cdot \\text{BCE} + 0.5 \\cdot \\text{Dice}$) and AdamW optimizer ($\\text{lr} = 10^{-4}$, $\\text{weight\\\_decay} = 10^{-4}$)\[cite: 1, 8]:

\- Baseline: Otsu (NDWI): Global IoU 0.9851, Global Dice 0.9925\[cite: 1, 5]

\- Model: U-Net (ResNet-34, 7 channels): Global IoU 0.9855, Global Dice 0.9927\[cite: 1, 5]



Key qualitative findings from validation tiles:

\- Noise suppression: Otsu produces isolated false-negative clusters in open water due to spectral variation from phytoplankton and shallow bottoms, whereas U-Net maintains complete spatial continuity\[cite: 1, 5, 9].

\- Edge super-resolution: SCL ground truth carries discrete 20 m nearest-neighbor block artifacts\[cite: 1, 4, 5]. U-Net leverages high-frequency 10 m skip connections from shallow encoder stages to output smooth, physically plausible shorelines consistent with native RGB imagery\[cite: 1, 4, 7, 9].



\## Visual Comparison

!\[Validation Results](outputs/figures/evaluation\_results.png)



\## Full Report

A complete technical report in Polish is available in this repository (`Sentinel-2\_U-Net.docx`)\[cite: 11].



\## Setup \& Run

```bash

pip install -r requirements.txt

python download\_data.py

python train.py

python evaluate.py

