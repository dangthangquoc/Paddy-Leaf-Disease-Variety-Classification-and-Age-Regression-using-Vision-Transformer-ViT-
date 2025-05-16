# COSC2753_A2_MachineLearning | Paddy Leaf Disease/Variety Classification and Age Regression using Vision Transformer (ViT)

This project focuses on classifying paddy leaf diseases, variety and age regression using a Vision Transformer (ViT) model. The model is trained on images of 10 different classes of paddy leaf conditions, including both healthy and diseased states. The dataset is PaddyDisease_datasets, you can download the dataset here: https://www.kaggle.com/datasets/azimnahid/paddydisease-datasets

## 🗂️ Project Structure
```
paddy_disease_classification/
├── paddy_models/ 
│ └── vit_model.keras
├── test_images/ 
│ └── *.jpg / *.png / ...
├── train_images/ 
│ └── class_name_1/
│ └── class_name_2/
│ └── ...
├── environment.yaml 
├── prediction_submision.csv
├── 
├── 
├── README.md
└── 
```



## 📋 Class Labels

The model predicts the following paddy conditions:
- `bacterial_leaf_blight`
- `bacterial_leaf_streak`
- `bacterial_panicle_blight`
- `blast`
- `brown_spot`
- `dead_heart`
- `downy_mildew`
- `hispa`
- `normal`
- `tungro`
  
## ✅ Prerequisites

Before running the code, ensure that you have the following installed:

- Python 
- Essential libraries:
  - `tensorflow` ≥ 2.15
  - `jupyter`
  - `matplotlib`
  - `keras` = 2.15
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tqdm`
  - `pillow`

These dependencies are included in the `environment.yaml` file. After setting up the environment (see below), you can verify that all required libraries are installed and available.

### ✅ Verify Installation

To ensure all packages are correctly installed, activate your environment and run the following commands in Conda:

```bash
conda activate vit_paddy_classification
```
```python

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import tqdm
from PIL import Image

print("✅ All required libraries are successfully installed!")

```

### 🛠️ Environment Setup

You can run this project using **Conda**

1. Make sure you have Miniconda or Anaconda installed.

2. Create and run environment:
```bash
conda env create -f environment.yaml
conda activate vit_paddy_classification
```

## 💼 Running the Code
Follow these steps to run the code and generate predictions:
1. **Prepare the Data**:
   - Ensure `train_images` and `test_images` folder are in the same directory as `.py`.
   - 

2. **Run the Script**:
   Execute the script:
   ```bash
   python .py
   ```

3. **Expected Runtime**:
   - The script takes approximately ?? minutes to run on a standard machine (8-core CPU, 16GB RAM).

## Output




