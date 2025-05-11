

# Plot Learning Curves
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_curves(history)

# Generate Predictions for Test Set
def create_test_dataset(test_path):
    """Create dataset for test images"""
    test_files = []
    test_ids = []
    
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(test_path, img_name)
            test_files.append(img_path)
            test_ids.append(img_name)
    
    # Create dataset with dummy labels (not used)
    dataset = tf.data.Dataset.from_tensor_slices((test_files, [0] * len(test_files)))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, test_ids

# Create test dataset
print("Creating test dataset...")
test_pred_dataset, test_image_ids = create_test_dataset(TEST_IMG_PATH)

# Generate predictions
print("Generating predictions...")
predictions = vit_variety_classifier.predict(test_pred_dataset)
predicted_variety_indices = np.argmax(predictions, axis=1)
predicted_varieties = variety_encoder.inverse_transform(predicted_variety_indices)

# Create submission dataframe
submission_df = pd.DataFrame({
    'image_id': test_image_ids,
    'variety': predicted_varieties
})

# Save predictions
submission_df.to_csv('variety_predictions.csv', index=False)
print("Predictions saved to 'variety_predictions.csv'")

# Create a more detailed submission file
confidence_df = pd.DataFrame({
    'image_id': test_image_ids,
    'variety': predicted_varieties,
    'confidence': np.max(predictions, axis=1)
})

# Add top 3 predictions for each image
for i in range(3):
    top_n_indices = np.argsort(predictions, axis=1)[:, -(i+1)]
    confidence_df[f'variety_top_{i+1}'] = variety_encoder.inverse_transform(top_n_indices)
    confidence_df[f'confidence_top_{i+1}'] = np.sort(predictions, axis=1)[:, -(i+1)]

confidence_df.to_csv('variety_predictions_detailed.csv', index=False)
print("Detailed predictions saved to 'variety_predictions_detailed.csv'")

# Model evaluation and analysis
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions for validation set
print("Generating validation predictions...")
val_predictions = []
val_true_labels = []

for batch in test_dataset:
    images, labels = batch
    preds = vit_variety_classifier.predict(images, verbose=0)
    val_predictions.extend(np.argmax(preds, axis=1))
    val_true_labels.extend(labels.numpy())

# Create classification report
print("\nClassification Report:")
print(classification_report(val_true_labels, val_predictions, 
                          target_names=variety_encoder.classes_))

# Create confusion matrix
cm = confusion_matrix(val_true_labels, val_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=variety_encoder.classes_,
            yticklabels=variety_encoder.classes_)
plt.title('Confusion Matrix for Variety Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nTraining completed successfully!")