import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, DepthwiseConv2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

# Define paths
TRAIN_IMAGES_PATH = r"D:\\COSC2753_A2_MachineLearning\\train_images"
TEST_IMAGES_PATH = r"D:\\COSC2753_A2_MachineLearning\\test_images"
META_TRAIN_CSV = r"D:\\COSC2753_A2_MachineLearning\\meta_train.csv"
OUTPUT_DIR = r"D:\\COSC2753_A2_MachineLearning\\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Exploratory Data Analysis (EDA)
def perform_eda(meta_train_csv):
    meta_df = pd.read_csv(meta_train_csv)
    images = []
    labels = []
    varieties = []
    ages = []
    for _, row in meta_df.iterrows():
        image_id = row['image_id']
        label = row['label']
        variety = row['variety']
        age = row['age']
        image_path = os.path.join(TRAIN_IMAGES_PATH, label, image_id)
        if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(image_path)
            labels.append(label)
            varieties.append(variety)
            ages.append(age)
        else:
            print(f"Warning: Image not found or invalid: {image_path}")

    data = pd.DataFrame({
        'image': images,
        'label': labels,
        'variety': varieties,
        'age': ages
    })

    print("Total images:", len(data))
    print("\nLabel distribution (disease):")
    print(data['label'].value_counts())
    print("\nVariety distribution:")
    print(data['variety'].value_counts())
    print("\nAge statistics:")
    print(data['age'].describe())

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.countplot(x='label', data=data)
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.subplot(1, 3, 2)
    sns.countplot(x='variety', data=data)
    plt.title('Variety Distribution')
    plt.xticks(rotation=45)
    plt.subplot(1, 3, 3)
    sns.histplot(data['age'], bins=20)
    plt.title('Age Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distributions.png'))
    plt.close()

    plt.figure(figsize=(15, 10))
    unique_labels = data['label'].unique()[:10]
    for i, label in enumerate(unique_labels):
        sample_image = data[data['label'] == label]['image'].iloc[0]
        img = Image.open(sample_image)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"{label}\nVariety: {data[data['label'] == label]['variety'].iloc[0]}\nAge: {data[data['label'] == label]['age'].iloc[0]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
    plt.close()

    return data

# 2. Data Preparation (Manual Image Loading)
def prepare_data(data):
    strat = data['label']
    train_df, valid_df = train_test_split(data, train_size=0.8, shuffle=True, random_state=123, stratify=strat)

    print("Train set size:", len(train_df))
    print("Validation set size:", len(valid_df))

    batch_size = 32
    img_size = (224, 224)

    def load_and_preprocess_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array

    # Load and preprocess train images
    train_images = np.array([load_and_preprocess_image(path) for path in train_df['image'].values])
    valid_images = np.array([load_and_preprocess_image(path) for path in valid_df['image'].values])

    # Encode labels and varieties
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    variety_encoder = LabelEncoder()

    train_labels = label_encoder.fit_transform(train_df['label'])
    valid_labels = label_encoder.transform(valid_df['label'])
    train_varieties = variety_encoder.fit_transform(train_df['variety'])
    valid_varieties = variety_encoder.transform(valid_df['variety'])

    train_labels = tf.keras.utils.to_categorical(train_labels)
    valid_labels = tf.keras.utils.to_categorical(valid_labels)
    train_varieties = tf.keras.utils.to_categorical(train_varieties)
    valid_varieties = tf.keras.utils.to_categorical(valid_varieties)

    # Prepare age data
    train_ages = train_df['age'].values
    valid_ages = valid_df['age'].values

    # Load test images
    test_images = [os.path.join(TEST_IMAGES_PATH, f"{i:06d}.jpg") for i in range(200001, 203470)
                   if os.path.exists(os.path.join(TEST_IMAGES_PATH, f"{i:06d}.jpg"))]
    test_images = np.array([load_and_preprocess_image(path) for path in test_images])

    return {
        'train_images': train_images,
        'valid_images': valid_images,
        'train_labels': train_labels,
        'valid_labels': valid_labels,
        'train_varieties': train_varieties,
        'valid_varieties': valid_varieties,
        'train_ages': train_ages,
        'valid_ages': valid_ages,
        'test_images': test_images,
        'train_df': train_df,
        'valid_df': valid_df,
        'label_encoder': label_encoder,
        'variety_encoder': variety_encoder
    }

# 3. Build Models
def build_vgg16_classification(num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_mobilenetv2_classification(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(BatchNormalization())

    def mobilenet_block(filters, strides=(1, 1)):
        block = Sequential([
            Conv2D(filters * 6, (1, 1), padding='same', activation='relu'),
            BatchNormalization(),
            DepthwiseConv2D((3, 3), strides=strides, padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(filters, (1, 1), padding='same', activation='relu'),
            BatchNormalization()
        ])
        return block

    model.add(mobilenet_block(16))
    model.add(mobilenet_block(24, strides=(2, 2)))
    model.add(mobilenet_block(32))
    model.add(mobilenet_block(64, strides=(2, 2)))
    model.add(mobilenet_block(96))
    model.add(mobilenet_block(160, strides=(2, 2)))
    model.add(mobilenet_block(320))
    model.add(Conv2D(1280, (1, 1), padding='same', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_vgg16_regression():
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    return model

def build_simple_cnn_regression():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    return model

# 4. Train Model
def train_model(model, train_data, valid_data, output_dir, task, model_name, regression=False):
    if regression:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(output_dir, f'best_model_{task}_{model_name}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]

    if task == 'age':
        history = model.fit(
            train_data['images'],
            train_data['ages'],
            validation_data=(valid_data['images'], valid_data['ages']),
            batch_size=32,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            train_data['images'],
            train_data['labels'],
            validation_data=(valid_data['images'], valid_data['labels']),
            batch_size=32,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )

    return history

# 5. Evaluate Models
def evaluate_classification_model(model, valid_images, valid_labels, class_indices, output_dir, task, model_name):
    valid_loss, valid_accuracy = model.evaluate(valid_images, valid_labels)
    print(f"{model_name} - Validation {task} Loss: {valid_loss:.4f}")
    print(f"{model_name} - Validation {task} Accuracy: {valid_accuracy:.4f}")

    predictions = model.predict(valid_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(valid_labels, axis=1)
    class_labels = list(class_indices.keys())

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{model_name} - {task} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f'{task}_confusion_matrix_{model_name}.png'))
    plt.close()

    report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    print(f"\n{model_name} - {task} Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    if report:
        report_df = pd.DataFrame(report).transpose()
        try:
            report_df.to_csv(os.path.join(output_dir, f'{task}_classification_report_{model_name}.csv'))
        except Exception as e:
            print(f"Error saving classification report to CSV: {e}")
    else:
        print("Warning: Classification report is empty, skipping CSV save.")

    return {
        'loss': valid_loss,
        'accuracy': valid_accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def evaluate_regression_model(model, valid_images, valid_ages, output_dir, task, model_name):
    valid_loss, valid_mae = model.evaluate(valid_images, valid_ages)
    print(f"{model_name} - Validation {task} Loss (MSE): {valid_loss:.4f}")
    print(f"{model_name} - Validation {task} MAE: {valid_mae:.4f}")

    predictions = model.predict(valid_images)
    predictions = predictions.flatten()  # Đảm bảo predictions là 1D array
    true_values = valid_ages

    # Tính thêm MAE và MSE bằng sklearn.metrics
    sk_mae = mean_absolute_error(true_values, predictions)
    sk_mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    print(f"{model_name} - Validation {task} Sklearn MAE: {sk_mae:.4f}")
    print(f"{model_name} - Validation {task} Sklearn MSE: {sk_mse:.4f}")
    print(f"{model_name} - Validation {task} R² Score: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'{model_name} - Age Predictions vs True Values')
    plt.savefig(os.path.join(output_dir, f'{task}_predictions_{model_name}.png'))
    plt.close()

    return {
        'loss': valid_loss,
        'mae': valid_mae,
        'sklearn_mae': sk_mae,
        'sklearn_mse': sk_mse,
        'r2': r2
    }

# 6. Generate Submission
def generate_submission(label_model, variety_model, age_model, test_images, label_encoder, variety_encoder, output_dir):
    image_ids = [f"{i:06d}.jpg" for i in range(200001, 203470)]
    valid_image_ids = [img_id for img_id in image_ids if os.path.exists(os.path.join(TEST_IMAGES_PATH, img_id))]

    label_predictions = label_model.predict(test_images)
    label_classes = np.argmax(label_predictions, axis=1)
    label_names = label_encoder.inverse_transform(label_classes)

    variety_predictions = variety_model.predict(test_images)
    variety_classes = np.argmax(variety_predictions, axis=1)
    variety_names = variety_encoder.inverse_transform(variety_classes)

    age_predictions = age_model.predict(test_images)
    age_predictions = np.round(age_predictions).astype(int)

    submission_df = pd.DataFrame({
        'image_id': valid_image_ids,
        'label': label_names,
        'variety': variety_names,
        'age': age_predictions.flatten()
    })

    full_submission_df = pd.DataFrame({'image_id': image_ids})
    full_submission_df = full_submission_df.merge(submission_df, on='image_id', how='left')
    full_submission_df.fillna({'label': '', 'variety': '', 'age': ''}, inplace=True)

    submission_path = os.path.join(output_dir, 'submission.csv')
    full_submission_df.to_csv(submission_path, index=False)

    print(f"Submission file saved to: {submission_path}")
    print(f"Total entries in submission: {len(full_submission_df)}")
    print("\nSample submission:")
    print(full_submission_df.head())

    return full_submission_df

# 7. Plot Training History
def plot_history(history, output_dir, task, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - {task} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - {task} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task}_training_history_{model_name}.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    data = perform_eda(META_TRAIN_CSV)
    data_dict = prepare_data(data)
    train_df = data_dict['train_df']
    valid_df = data_dict['valid_df']

    print("\nTraining Label Models...")
    num_label_classes = data_dict['train_labels'].shape[1]

    label_vgg = build_vgg16_classification(num_label_classes)
    label_vgg_history = train_model(
        label_vgg,
        {'images': data_dict['train_images'], 'labels': data_dict['train_labels']},
        {'images': data_dict['valid_images'], 'labels': data_dict['valid_labels']},
        OUTPUT_DIR, 'label', 'vgg16'
    )
    plot_history(label_vgg_history, OUTPUT_DIR, 'Label', 'VGG16')
    label_vgg_metrics = evaluate_classification_model(
        label_vgg, data_dict['valid_images'], data_dict['valid_labels'],
        data_dict['label_encoder'].classes_, OUTPUT_DIR, 'Label', 'VGG16'
    )

    label_mobile = build_mobilenetv2_classification(num_label_classes)
    label_mobile_history = train_model(
        label_mobile,
        {'images': data_dict['train_images'], 'labels': data_dict['train_labels']},
        {'images': data_dict['valid_images'], 'labels': data_dict['valid_labels']},
        OUTPUT_DIR, 'label', 'mobilenetv2'
    )
    plot_history(label_mobile_history, OUTPUT_DIR, 'Label', 'MobileNetV2')
    label_mobile_metrics = evaluate_classification_model(
        label_mobile, data_dict['valid_images'], data_dict['valid_labels'],
        data_dict['label_encoder'].classes_, OUTPUT_DIR, 'Label', 'MobileNetV2'
    )

    print("\nTraining Variety Models...")
    num_variety_classes = data_dict['train_varieties'].shape[1]

    variety_vgg = build_vgg16_classification(num_variety_classes)
    variety_vgg_history = train_model(
        variety_vgg,
        {'images': data_dict['train_images'], 'labels': data_dict['train_varieties']},
        {'images': data_dict['valid_images'], 'labels': data_dict['valid_varieties']},
        OUTPUT_DIR, 'variety', 'vgg16'
    )
    plot_history(variety_vgg_history, OUTPUT_DIR, 'Variety', 'VGG16')
    variety_vgg_metrics = evaluate_classification_model(
        variety_vgg, data_dict['valid_images'], data_dict['valid_varieties'],
        data_dict['variety_encoder'].classes_, OUTPUT_DIR, 'Variety', 'VGG16'
    )

    variety_mobile = build_mobilenetv2_classification(num_variety_classes)
    variety_mobile_history = train_model(
        variety_mobile,
        {'images': data_dict['train_images'], 'labels': data_dict['train_varieties']},
        {'images': data_dict['valid_images'], 'labels': data_dict['valid_varieties']},
        OUTPUT_DIR, 'variety', 'mobilenetv2'
    )
    plot_history(variety_mobile_history, OUTPUT_DIR, 'Variety', 'MobileNetV2')
    variety_mobile_metrics = evaluate_classification_model(
        variety_mobile, data_dict['valid_images'], data_dict['valid_varieties'],
        data_dict['variety_encoder'].classes_, OUTPUT_DIR, 'Variety', 'MobileNetV2'
    )

    print("\nTraining Age Models...")
    age_vgg = build_vgg16_regression()
    age_vgg_history = train_model(
        age_vgg,
        {'images': data_dict['train_images'], 'ages': data_dict['train_ages']},
        {'images': data_dict['valid_images'], 'ages': data_dict['valid_ages']},
        OUTPUT_DIR, 'age', 'vgg16', regression=True
    )
    plot_history(age_vgg_history, OUTPUT_DIR, 'Age', 'VGG16')
    age_vgg_metrics = evaluate_regression_model(
        age_vgg, data_dict['valid_images'], data_dict['valid_ages'],
        OUTPUT_DIR, 'Age', 'VGG16'
    )

    age_simple = build_simple_cnn_regression()
    age_simple_history = train_model(
        age_simple,
        {'images': data_dict['train_images'], 'ages': data_dict['train_ages']},
        {'images': data_dict['valid_images'], 'ages': data_dict['valid_ages']},
        OUTPUT_DIR, 'age', 'simple_cnn', regression=True
    )
    plot_history(age_simple_history, OUTPUT_DIR, 'Age', 'Simple_CNN')
    age_simple_metrics = evaluate_regression_model(
        age_simple, data_dict['valid_images'], data_dict['valid_ages'],
        OUTPUT_DIR, 'Age', 'Simple_CNN'
    )

    best_label_model = label_vgg if label_vgg_metrics['accuracy'] > label_mobile_metrics['accuracy'] else label_mobile
    best_label_indices = data_dict['label_encoder'].classes_
    print(f"Best Label Model: {'VGG16' if best_label_model == label_vgg else 'MobileNetV2'} (Accuracy: {max(label_vgg_metrics['accuracy'], label_mobile_metrics['accuracy']):.4f})")

    best_variety_model = variety_vgg if variety_vgg_metrics['accuracy'] > variety_mobile_metrics['accuracy'] else variety_mobile
    best_variety_indices = data_dict['variety_encoder'].classes_
    print(f"Best Variety Model: {'VGG16' if best_variety_model == variety_vgg else 'MobileNetV2'} (Accuracy: {max(variety_vgg_metrics['accuracy'], variety_mobile_metrics['accuracy']):.4f})")

    best_age_model = age_vgg if age_vgg_metrics['mae'] < age_simple_metrics['mae'] else age_simple
    print(f"Best Age Model: {'VGG16' if best_age_model == age_vgg else 'Simple_CNN'} (MAE: {min(age_vgg_metrics['mae'], age_simple_metrics['mae']):.4f})")

    submission_df = generate_submission(
        best_label_model, best_variety_model, best_age_model,
        data_dict['test_images'], data_dict['label_encoder'], data_dict['variety_encoder'], OUTPUT_DIR
    )

    print("Training, evaluation, and submission completed. Check output directory for plots, models, and submission file.")