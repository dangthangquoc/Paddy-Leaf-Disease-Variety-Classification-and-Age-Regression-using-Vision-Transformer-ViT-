# COSC2753_A2_MachineLearning | Paddy Leaf Disease/Variety Classification and Age Regression using Vision Transformer (ViT)

This project focuses on classifying paddy leaf diseases, variety and age regression using a Vision Transformer (ViT) model. The model is trained on images of 10 different classes of paddy leaf conditions, including both healthy and diseased states. The dataset is PaddyDisease_datasets, you can download the dataset here: https://www.kaggle.com/datasets/azimnahid/paddydisease-datasets

## 🗂️ Project Structure
```
paddy_disease_classification/
├── task3_load_model.ipynb
├── task2_load_model.ipynb
├── task1_load_model.ipynb
├── prediction_submission.csv
├── paddy_doctor_app.py
├── meta_train.csv
├── generate_prediction.ipynb
├── exploratory_data_analysis.ipynb
├── environment.yaml
├── age_stats_kfold.json
├── train_scripts/
│ ├── age_regressor_training.ipynb
│ ├── disease_classification_training.ipynb
│ └── variety_classification_training.ipynb
├── train_images/
├── test_images/
├── disease_label_encoder.joblib
├── variety_label_encoder.joblib
├── detailed_prediction/
│ ├── age_predictions_submission.csv
│ ├── disease_predictions.csv
│ ├── variety_predictions.csv
│ ├── disease_predictions_detailed.csv
│ └── variety_predictions_detailed.csv
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




