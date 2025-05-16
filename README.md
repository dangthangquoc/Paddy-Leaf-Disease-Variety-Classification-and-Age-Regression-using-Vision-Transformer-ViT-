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
├── age_regressor_training.ipynb
├── disease_classification_training.ipynb
├── variety_classification_training.ipynb
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

## 📄 File Descriptions

Here is a brief overview of the files included in this project:

- `paddy_doctor_app.py` — Main app file for launching the UI to make real-time predictions using trained models.
- `generate_prediction.ipynb` — Script to generate and compile final predictions for all tasks.
- `exploratory_data_analysis.ipynb` — Visualizes dataset statistics and explores class distributions, image samples, and other insights.
- `environment.yaml` — Conda environment definition file including all required dependencies.
- `meta_train.csv` — Metadata file containing image paths and labels used for training and evaluation.
- `age_stats_kfold.json` — Stores fold-wise mean and standard deviation values used for normalization in age regression.
- `age_regressor_training.ipynb` — Notebook to train the age prediction model using regression techniques and K-Fold cross-validation.
- `disease_classification_training.ipynb` — Notebook for training a classification model to detect paddy leaf diseases.
- `variety_classification_training.ipynb` — Notebook for training a model to classify the variety of the paddy plant.
- `task1_load_model.ipynb` — Loads the trained model for Task 1 (Disease classification) and generates predictions.
- `task2_load_model.ipynb` — Loads the trained model for Task 2 (Variety classification) and generates predictions.
- `task3_load_model.ipynb` — Loads the trained model for Task 3 (Age regression) and generates predictions.
- `prediction_submission.csv` — Final combined predictions submitted for evaluation (typically on Kaggle or a leaderboard).
- `train_images/` — Folder containing training images used for model development.
- `test_images/` — Folder containing test images used for evaluation and submission.
- `disease_label_encoder.joblib` — Serialized encoder for converting disease class names to numerical labels and back.
- `variety_label_encoder.joblib` — Serialized encoder for converting variety class names to numerical labels and back.
- `detailed_prediction/` — Folder containing detailed prediction outputs for each task:
  - `age_predictions_submission.csv` — Final predicted ages for the test images.
  - `disease_predictions.csv` — Basic disease predictions for test set.
  - `variety_predictions.csv` — Basic variety predictions for test set.
  - `disease_predictions_detailed.csv` — Disease predictions with probabilities/confidences or metadata.
  - `variety_predictions_detailed.csv` — Variety predictions with additional details or scores.
  
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

Follow these steps to run the application and generate predictions:

### 1. 📁 Prepare the Data

Ensure the following structure exists in the same directory as `paddy_doctor_app.py`:
```
├── paddy_doctor_app.py
├── paddy_models
│ ├── vit_label_weights.weights.h5
│ ├── vit_variety_weights.weights.h5
│ └── kfold_models
│ ├── best_vit_age_model_fold_1.weights.h5
│ ├── best_vit_age_model_fold_2.weights.h5
│ └── best_vit_age_model_fold_3.weights.h5
├── encoder
```


### 2. 🚀 Run the Application

Execute the script with:

```bash
python paddy_doctor_app.py
```

### 3. ⏱️ Expected Runtime
The script takes approximately 7 seconds to run on a standard machine (8-core CPU, 16GB RAM).

## 🖼️ Output
You should see the application UI:

<p align="center"> <img src="https://github.com/user-attachments/assets/bc3223c7-7c17-4f17-b4de-134cc3df2ba1" alt="App UI" width="600"/> </p>


## 🔁 Reproduce Training & Prediction Process
### 1. 📁 Prepare the Data
Place the following in the same directory as the notebook:
```
.
├── train_images/
├── test_images/
├── meta_train.csv
├── {task_name}_{task_type}_training.ipynb
```
Where: 
- `{task_name}`: `disease`, `variety`, or `age`
- `{task_type}`: `classification` or `regressor`

### 2. 📓 Execute Training
Open and run the appropriate Jupyter notebook:

```bash
{task_name}_{task_type}_training.ipynb
```

### 3. 🧠 Run Predictions
Run the prediction notebook:

```bash
{task_number}_load_predict.ipynb
```
Where:
`{task_number}`: `1`, `2`, or `3` (corresponds to the specific task)

After completing the notebook, you should see prediction files generated like the following:

<p align="center"> <img src="https://github.com/user-attachments/assets/c6d61fe5-8c56-42b8-a0b5-513d3883e802" alt="Prediction Output" width="600"/> </p>




