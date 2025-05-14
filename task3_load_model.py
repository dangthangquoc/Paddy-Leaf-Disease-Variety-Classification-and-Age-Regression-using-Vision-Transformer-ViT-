import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from PIL import Image
import pandas as pd
from pathlib import Path

# Set random seeds for reproducibility
import random
random.seed(45)
tf.random.set_seed(45)
np.random.seed(45)

# Define constants (same as in training script)
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_units = [2048, 1024]

# MLP helper function
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Define custom layers
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
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
        
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
    @classmethod
    def from_config(cls, config):
        patch_size = config.pop("patch_size")
        return cls(patch_size=patch_size, **config)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        num_patches = config.pop("num_patches")
        projection_dim = config.pop("projection_dim")
        return cls(num_patches=num_patches, projection_dim=projection_dim, **config)

# Function to create the ViT model
def create_vit_regressor(input_images=None):
    # Create data augmentation inside model
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    
    # Adapt normalization layer if input images are provided
    if input_images is not None:
        data_augmentation.layers[0].adapt(input_images)
    
    inputs = layers.Input(shape=(256, 256, 3))
    
    # Augment data
    augmented = data_augmentation(inputs)
    
    # Create patches
    patches = Patches(patch_size)(augmented)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Output layer for regression (single neuron for age)
    output = layers.Dense(1, activation='linear')(features)
    
    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=output)
    return model

# Function to load the ensemble models
def load_ensemble_models(models_dir, k_folds=3):
    ensemble_models = []
    
    # Load age statistics
    with open(os.path.join(models_dir, 'age_stats_kfold.json'), 'r') as f:
        age_stats = json.load(f)
        average_age_mean = age_stats['mean']
        average_age_std = age_stats['std']
    
    # For testing purposes, if no stats file exists, use these defaults
    if not os.path.exists(os.path.join(models_dir, 'age_stats_kfold.json')):
        print("Warning: Could not find age statistics file. Using default values.")
        average_age_mean = 64.0436244835207  # Example value, should be replaced with actual mean
        average_age_std = 8.9582253420494    # Example value, should be replaced with actual std
    
    # Create and load each fold model
    for fold in range(1, k_folds + 1):
        # Create a fresh model
        model = create_vit_regressor()
        
        # Compile model (not strictly necessary for inference but good practice)
        optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='mean_absolute_error',
            metrics=['mae', 'mse']
        )
        
        # Load weights
        weights_path = os.path.join(models_dir, f'paddy_models/kfold_models/best_vit_age_model_fold_{fold}.weights.h5')
        
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"Loaded weights for fold {fold} from {weights_path}")
            
            # Load individual fold stats if available, otherwise use average
            # You could implement fold-specific stats if needed
            fold_mean = average_age_mean
            fold_std = average_age_std
            
            ensemble_models.append((model, fold_mean, fold_std))
        else:
            print(f"Warning: Could not find weights for fold {fold} at {weights_path}")
    
    if not ensemble_models:
        raise FileNotFoundError("No model weights could be loaded. Please check the paths.")
        
    return ensemble_models

# Function to preprocess images
def preprocess_images(image_paths):
    preprocessed_images = []
    image_ids = []
    
    for img_path in image_paths:
        if isinstance(img_path, str) and os.path.exists(img_path):
            # Extract image id
            img_id = Path(img_path).stem
            image_ids.append(img_id)
            
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image = image.resize((256, 256))
            image = np.array(image)
            preprocessed_images.append(image)
    
    return np.array(preprocessed_images), image_ids

# Function to make predictions with the ensemble
def predict_paddy_age(image_paths, ensemble_models):
    # Preprocess images
    preprocessed_images, image_ids = preprocess_images(image_paths)
    
    if len(preprocessed_images) == 0:
        print("No valid images were provided for prediction.")
        return None
    
    # Make predictions with each model in the ensemble
    all_predictions = []
    
    for model, age_mean, age_std in ensemble_models:
        predictions_norm = model.predict(preprocessed_images, verbose=0)
        print(f"Predictions (normalized) for model: {predictions_norm.flatten()}")
        predictions_original = predictions_norm.flatten() * age_std + age_mean
        all_predictions.append(predictions_original)
    
    # Average predictions across all models in the ensemble
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'predicted_age': ensemble_predictions,
        'predicted_age_rounded': np.round(ensemble_predictions).astype(int)
    })
    
    return results_df

# Function to create a submission file
def create_submission(predictions_df, output_path):
    submission_df = pd.DataFrame({
        'image_id': predictions_df['image_id'],
        'age': predictions_df['predicted_age_rounded'].astype(str)
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    
    return submission_df

# Main function to demonstrate usage
def main():
    # Set the path to your models directory
    models_dir = os.getcwd()  # Adjust as needed
    
    # Load ensemble models
    print("Loading ensemble models...")
    ensemble_models = load_ensemble_models(models_dir, k_folds=3)
    
    # Example: predict on test images
    test_dir = os.path.join(models_dir, 'test_images')
    
    if os.path.exists(test_dir):
        print(f"Found test directory: {test_dir}")
        # Get all image files from the test directory
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            print(f"Making predictions on {len(test_images)} test images...")
            predictions_df = predict_paddy_age(test_images, ensemble_models)
            
            # Create submission file
            output_path = os.path.join(models_dir, 'age_predictions_submission.csv')
            create_submission(predictions_df, output_path)
        else:
            print("No image files found in the test directory.")
    else:
        print(f"Test directory not found: {test_dir}")
        print("Please provide the path to your test images to make predictions.")

if __name__ == "__main__":
    main()