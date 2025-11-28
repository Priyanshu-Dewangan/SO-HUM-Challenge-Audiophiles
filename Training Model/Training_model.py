import os
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tf2onnx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
dataset_path = "C:\\Users\\priya\\Downloads\\So Hum Challenge Dataset" 

n_mfcc = 13
n_fft = 2048
hop_length = 512
fixed_mfcc_length = 1000

# --- 1. SETUP ---
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path not found: {dataset_path}")
else:
    os.chdir(dataset_path)

# Load Metadata
if os.path.exists('Metadata.csv'):
    metadata_df = pd.read_csv('Metadata.csv')
    print("Metadata loaded successfully.")
else:
    print("Error: Metadata.csv not found.")
    exit()

class_map = {'HS': 'Heart sound', 'LS': 'Lungs sound', 'Noise': 'Noise'}
label_to_folder = {v: k for k, v in class_map.items()}

# --- 2. DATA LOADING ---
print("Loading and processing audio files...")
audio_data = []
labels = []

for index, row in metadata_df.iterrows():
    # Use your CSV column names
    file_name = row['new_filename']
    label_name = row['class']
    
    subdir = label_to_folder.get(label_name)
    if not subdir: continue

    file_path = os.path.join(dataset_path, subdir, file_name)

    if os.path.exists(file_path):
        try:
            # CRITICAL: 16kHz Sampling Rate
            y, sr = librosa.load(file_path, sr=16000) 

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            audio_data.append(mfccs)
            labels.append(label_name)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print(f"Loaded {len(audio_data)} files.")

if len(audio_data) == 0:
    print("Error: No files loaded.")
    exit()

# --- 3. PRE-PROCESSING ---
processed_mfccs = []
for mfccs in audio_data:
    if mfccs.shape[1] > fixed_mfcc_length:
        mfccs = mfccs[:, :fixed_mfcc_length]
    else:
        pad_width = fixed_mfcc_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    processed_mfccs.append(mfccs)

processed_mfccs = np.array(processed_mfccs)

# Transpose to (Samples, Time, Features)
X = np.transpose(processed_mfccs, (0, 2, 1))

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

# CRITICAL: Robust Scaling
N_train, T_train, F_dim = X_train.shape
N_test, T_test, _ = X_test.shape

scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, F_dim)
scaler.fit(X_train_flat)

X_train_scaled = scaler.transform(X_train_flat).reshape(N_train, T_train, F_dim)
X_test_scaled = scaler.transform(X_test.reshape(-1, F_dim)).reshape(N_test, T_test, F_dim)

# Save Scaler
joblib.dump(scaler, 'feature_scaler.joblib')
print("Saved feature_scaler.joblib")

# --- 4. MODEL TRAINING ---
input_shape = (fixed_mfcc_length, n_mfcc) 
num_classes = len(label_encoder.classes_)

model = Sequential()
model.add(Input(shape=(None, n_mfcc))) 

model.add(Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[checkpoint, early_stopping]
)

# --- 5. EVALUATION ---
best_model = load_model('best_model.keras')
test_loss, test_acc = best_model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc}")

y_pred_prob = best_model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert One-Hot to Indices for Metrics
y_true_indices = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_indices, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_indices, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_test_bin = label_binarize(y_true_indices, classes=range(num_classes))
if num_classes == 2: 
    y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

fpr = dict()
tpr = dict()
roc_auc = dict()

print("\nROC AUC Scores per Class:")
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"  {label_encoder.classes_[i]}: {roc_auc[i]:.4f}")

# --- 6. ONNX EXPORT (YOUR REQUESTED METHOD) ---
print("\nExporting to ONNX using tf2onnx.from_function...")

try:
    import tf2onnx
    best_model = tf.keras.models.load_model('best_model.keras')

    # Determine input dims from training tensors
    time_steps = int(X_train_scaled.shape[1])
    n_mfcc = int(X_train_scaled.shape[2])

    # Wrap model in a tf.function with concrete signature (None, T, F)
    input_sig = [tf.TensorSpec(shape=[None, time_steps, n_mfcc], dtype=tf.float32, name="input")]

    @tf.function(input_signature=input_sig)
    def model_fn(x):
        return best_model(x)

    # Convert using from_function and explicitly provide the input_signature
    tf2onnx.convert.from_function(
        model_fn,
        input_signature=input_sig,
        opset=13,
        output_path="model.onnx"
    )
    print('Exported ONNX model to model.onnx')

except Exception as e:
    print(f"ONNX export failed: {e}. Ensure tf2onnx is installed (pip install tf2onnx) and try again.")