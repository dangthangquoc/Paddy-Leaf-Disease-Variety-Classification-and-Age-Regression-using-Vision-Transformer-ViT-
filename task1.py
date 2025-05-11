import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model, Input
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
import warnings
import logging
import json
from glob import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")
random.seed(45)
np.random.seed(45)
tf.random.set_seed(45)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# **DEFINE MISSING CLASSES AND FUNCTIONS**

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, name=None):
        super(TransformerBlock, self).__init__(name=name)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None, return_attention=False):
        attn_output, attn_weights = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        if return_attention:
            return out2, attn_weights
        return out2

# **DATA LOADING FUNCTIONS**

def load_images_and_labels(train_images_path, meta_df):
    """Load images and create labels array"""
    images = []
    labels = []
    image_paths = []
    
    logger.info("Loading training images...")
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        image_id = row['image_id']
        label = row['label']
        
    
        image_path = os.path.join(train_images_path, label, f"{image_id}")
        
        if os.path.exists(image_path):
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            image_array = np.array(image) / 255.0
            
            images.append(image_array)
            labels.append(label)
            image_paths.append(image_path)
    
    # Convert labels to numeric
    unique_labels = sorted(meta_df['label'].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_index[label] for label in labels]
    
    return np.array(images), np.array(numeric_labels), unique_labels, image_paths

def load_test_images(test_images_path):
    """Load test images for prediction"""
    images = []
    image_paths = []
    
    logger.info("Loading test images...")
    for image_file in tqdm(os.listdir(test_images_path)):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_images_path, image_file)
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            image_array = np.array(image) / 255.0
            
            images.append(image_array)
            image_paths.append(image_file)
    
    return np.array(images), image_paths

# **EXPERIMENT MANAGER CLASS**

class ExperimentManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results_dir = f"results/{experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("model_evaluation", exist_ok=True)
        self.results = {}
    
    def log_experiment(self, model_name, metrics, config):
        self.results[model_name] = {
            'metrics': metrics,
            'config': config
        }
        
        # Save results to JSON
        with open(f"{self.results_dir}/results.json", 'w') as f:
            json.dump(self.results, f, indent=4)

# **DATASET ANALYSIS**

def analyze_dataset_comprehensive(meta_train_df):
    """Perform comprehensive dataset analysis for HD/DI requirements"""
    
    # Class imbalance analysis
    class_distribution = meta_train_df['label'].value_counts()
    imbalance_ratio = class_distribution.max() / class_distribution.min()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Class distribution
    class_distribution.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title(f'Class Distribution (Imbalance Ratio: {imbalance_ratio:.2f}:1)')
    axes[0,0].set_xlabel('Disease Class')
    axes[0,0].set_ylabel('Number of Images')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Variety distribution
    variety_distribution = meta_train_df['variety'].value_counts()
    variety_distribution.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Variety Distribution')
    axes[0,1].set_xlabel('Paddy Variety')
    axes[0,1].set_ylabel('Number of Images')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Age distribution
    axes[1,0].hist(meta_train_df['age'], bins=30, edgecolor='black')
    axes[1,0].set_title('Age Distribution')
    axes[1,0].set_xlabel('Age (days)')
    axes[1,0].set_ylabel('Frequency')
    
    # Plot 4: Class-Variety heatmap
    cross_table = pd.crosstab(meta_train_df['label'], meta_train_df['variety'])
    sns.heatmap(cross_table, cmap='YlOrRd', annot=True, fmt='d', ax=axes[1,1])
    axes[1,1].set_title('Class-Variety Distribution Heatmap')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'class_distribution': class_distribution,
        'imbalance_ratio': imbalance_ratio,
        'variety_distribution': variety_distribution,
        'age_stats': {
            'mean': meta_train_df['age'].mean(),
            'std': meta_train_df['age'].std(),
            'min': meta_train_df['age'].min(),
            'max': meta_train_df['age'].max()
        }
    }

# **ENHANCED VIT MODEL**

def create_enhanced_vit_classifier(num_classes, input_shape=(256, 256, 3)):
    """Create enhanced ViT classifier with better architecture"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.Rescale(1.0/255.0),
        layers.Resizing(72, 72),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.05),
        layers.RandomZoom(height_factor=0.3, width_factor=0.3),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.2)
    ])
    
    x = data_augmentation(inputs)
    
    # Create patches
    patches = Patches(patch_size=6)(x)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches=144, projection_dim=64)(patches)
    
    # Transformer blocks
    for i in range(8):
        encoded_patches = TransformerBlock(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            rate=0.1
        )(encoded_patches)
    
    # Classification head
    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Enhanced MLP head
    x = layers.Dense(2048, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes)(x)
    
    model = Model(inputs, outputs)
    
    return model

# **CROSS VALIDATION**

def stratified_cross_validation(X, y, num_classes, n_splits=5):
    """Perform stratified k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = []
    fold_models = []
    
    # Calculate class weights for the entire dataset
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and compile model for this fold
        model = create_enhanced_vit_classifier(num_classes=num_classes)
        
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        )
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/fold_{fold}_best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate fold
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Store results
        fold_results = {
            'fold': fold + 1,
            'best_val_accuracy': max(history.history['val_accuracy']),
            'final_val_accuracy': val_accuracy
        }
        
        cv_results.append(fold_results)
        fold_models.append(model)
        
        logger.info(f"Fold {fold + 1} completed - Val Accuracy: {val_accuracy:.4f}")
    
    return cv_results, fold_models

# **MODEL ENSEMBLE**

class ModelEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X, batch_size=32):
        """Ensemble prediction using voting"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, batch_size=batch_size)
            predictions.append(pred)
        
        # Average logits (before softmax)
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict_proba(self, X, batch_size=32):
        """Get probability predictions from ensemble"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, batch_size=batch_size)
            predictions.append(tf.nn.softmax(pred).numpy())
        
        # Average probabilities
        ensemble_proba = np.mean(predictions, axis=0)
        return ensemble_proba

# **COMPREHENSIVE EVALUATION**

def comprehensive_model_evaluation(model, X_test, y_test, class_names):
    """Comprehensive evaluation with advanced metrics and visualizations"""
    
    # Get predictions
    y_pred_logits = model.predict(X_test)
    y_pred_probs = tf.nn.softmax(y_pred_logits).numpy()
    y_pred = np.argmax(y_pred_logits, axis=1)
    
    # Create evaluation directory
    eval_dir = "model_evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{eval_dir}/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 2. Per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Save classification report
    with open(f'{eval_dir}/classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    # Visualize per-class metrics
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
    plt.title('Per-Class Performance Metrics')
    plt.xlabel('Disease Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{eval_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC curves for each class
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    for i, color in zip(range(len(class_names)), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{eval_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc
    }

# **MAIN TRAINING PIPELINE**

def train_enhanced_model(meta_train_path, train_images_path, test_images_path):
    """Main training pipeline with all HD/DI improvements"""
    
    # Initialize experiment manager
    experiment = ExperimentManager("enhanced_vit_task1")
    
    # Load metadata
    logger.info("Loading metadata...")
    meta_df = pd.read_csv(meta_train_path)
    
    # Comprehensive dataset analysis
    logger.info("Performing comprehensive dataset analysis...")
    dataset_analysis = analyze_dataset_comprehensive(meta_df)
    
    # Load images and labels
    logger.info("Loading training data...")
    X, y, class_names, image_paths = load_images_and_labels(train_images_path, meta_df)
    
    logger.info(f"Loaded {len(X)} training images with {len(class_names)} classes")
    logger.info(f"Class names: {class_names}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} images")
    logger.info(f"Test set: {len(X_test)} images")
    
    # Train with cross-validation
    logger.info("Starting stratified cross-validation...")
    cv_results, fold_models = stratified_cross_validation(
        X_train, y_train, num_classes=len(class_names), n_splits=5
    )
    
    # Create ensemble model
    logger.info("Creating ensemble model...")
    ensemble = ModelEnsemble(fold_models)
    
    # Comprehensive evaluation on test set
    logger.info("Performing comprehensive evaluation...")
    evaluation_results = comprehensive_model_evaluation(
        ensemble, X_test, y_test, class_names
    )
    
    # Load test images
    logger.info("Loading test images for final prediction...")
    test_images, test_image_files = load_test_images(test_images_path)
    
    # Generate final predictions for submission
    logger.info("Generating final predictions...")
    test_predictions = ensemble.predict_proba(test_images)
    predicted_labels = [class_names[np.argmax(pred)] for pred in test_predictions]
    confidence_scores = [np.max(pred) for pred in test_predictions]
    
    # Create submission file
    submission_df = pd.DataFrame({
        'image_id': test_image_files,
        'label': predicted_labels,
        'confidence': confidence_scores
    })
    
    submission_df.to_csv('enhanced_submission.csv', index=False)
    logger.info("Submission file created: enhanced_submission.csv")
    
    # Save experiment results
    experiment.log_experiment(
        "enhanced_ensemble_vit",
        evaluation_results,
        {
            'architecture': 'Enhanced ViT with ensemble',
            'cross_validation_folds': 5,
            'class_weights': 'balanced',
            'augmentation': 'Advanced data augmentation',
            'num_classes': len(class_names),
            'cv_results': cv_results
        }
    )
    
    logger.info("Training pipeline completed successfully!")
    
    return experiment, ensemble, evaluation_results

# **MAIN EXECUTION**

if __name__ == "__main__":
    HOME_PATH = os.getcwd() + "/"
    
    # Run the complete training pipeline
    experiment, model, results = train_enhanced_model(
        meta_train_path= HOME_PATH + 'meta_train.csv',
        train_images_path= HOME_PATH +'train_images/',
        test_images_path= HOME_PATH +'test_images/'
    )
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Results saved in: {experiment.results_dir}")
    print(f"Models saved in: models/")
    print(f"Evaluation plots saved in: model_evaluation/")
    print(f"Final submission saved as: enhanced_submission.csv")
    print("="*50)