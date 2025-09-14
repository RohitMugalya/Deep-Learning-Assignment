import os
import cv2
import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.metrics import normalized_root_mse
from piq import niqe, brisque, piqe
import torch

# Folders containing outputs of different methods
output_folders = {
    "HistogramEQ": "traditional_model/test_output/histogram_eq",
    "CLAHE": "traditional_model/test_output/clahe",
    "Gamma": "traditional_model/test_output/gamma_correction",
    "RetinexSSR": "traditional_model/test_output/retinex_ssr",
    "RetinexMSR": "traditional_model/test_output/retinex_msr",
    "LIME": "traditional_model/test_output/lime",
    "Exposure": "traditional_model/test_output/exposure_correction",
    "AdaptiveGamma": "traditional_model/test_output/adaptive_gamma",
    "DL_Model": "test_output"
}

results = []

for method, folder in output_folders.items():
    niqe_scores, brisque_scores, piqe_scores = [], [], []
    
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        try:
            niqe_scores.append(niqe(tensor_img).item())
            brisque_scores.append(brisque(tensor_img).item())
            piqe_scores.append(piqe(tensor_img).item())
        except Exception as e:
            print(f"Skipping {file}: {e}")
    
    results.append({
        "Method": method,
        "NIQE": np.mean(niqe_scores) if niqe_scores else None,
        "BRISQUE": np.mean(brisque_scores) if brisque_scores else None,
        "PIQE": np.mean(piqe_scores) if piqe_scores else None,
    })

df = pd.DataFrame(results)
print(df)
df.to_csv("no_reference_metrics.csv", index=False)
