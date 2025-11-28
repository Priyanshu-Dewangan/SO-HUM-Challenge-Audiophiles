import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def load_audio_full(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load entire audio file at target sampling rate as mono.
    Returns a 1D float32 numpy array normalized to [-1, 1].
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if y.size == 0:
        return np.zeros((target_sr,), dtype=np.float32)
    # Normalize to [-1,1]
    max_abs = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    y = (y / max_abs).astype(np.float32)
    return y


def compute_features(y: np.ndarray, sr: int = 16000, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512, fixed_len: int = 1000, scaler=None) -> np.ndarray:
    """
    Pre-processing to MFCCs consistent with training.
    """
    # 1. Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    x = mfcc.T  # Shape: (Time, Features)
    
    # 2. Pad or Trim (To match the input signature of the ONNX model)
    if x.shape[0] > fixed_len:
        x = x[:fixed_len, :]
    elif x.shape[0] < fixed_len:
        pad = fixed_len - x.shape[0]
        x = np.pad(x, ((0, pad), (0, 0)), mode='constant')

    # 3. Apply Robust Scaler (UPDATED)
    # Since the scaler was fitted on (N*T, F), we can transform (T, F) directly.
    # This works for any audio length.
    if scaler is not None:
        x = scaler.transform(x)
        
    return x.astype(np.float32)


def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray):
    # y_prob: (N, num_classes)
    y_pred = np.argmax(y_prob, axis=1)
    # For binary AUC-ROC, handle num_classes==2; otherwise macro-average with One-vs-Rest
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, average='macro')),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro')),
        'roc_auc': float(auc) if auc == auc else None,
    }
    return y_pred, metrics


def main():
    parser = argparse.ArgumentParser(description="Audio classification ONNX inference")
    parser.add_argument('--data', required=True, help='Path to /test_input')
    parser.add_argument('--output', required=True, help='Path to /output')
    parser.add_argument('--model', default='/app/model.onnx', help='Path to ONNX model')
    parser.add_argument('--sr', type=int, default=16000, help='Target sampling rate')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs')
    parser.add_argument('--fixed_len', type=int, default=1000, help='Fixed number of frames for MFCC sequences')
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / 'test_meta_data.csv'
    raw_dir = data_dir / 'raw_audio'

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV at {meta_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw_audio directory at {raw_dir}")

    df = pd.read_csv(meta_path)
    
    # Robust Column Name Handling (Matching training logic)
    col_map = {c.lower(): c for c in df.columns}
    audio_col = col_map.get('audio_name') or col_map.get('filename') or df.columns[0]
    dir_col = col_map.get('audio_directory') or col_map.get('path') or df.columns[1]
    label_col = col_map.get('labels') or col_map.get('label') or df.columns[2]

    audio_names = df[audio_col].astype(str).tolist()
    
    # Correctly resolve paths
    audio_paths = [str((raw_dir / str(p)).resolve()) if not os.path.isabs(str(p)) else str(p) for p in df[dir_col]]
    
    y_true = df[label_col].values
    
    # Encode labels to numeric
    le = None
    if y_true.dtype.kind in {'U', 'S', 'O'}:
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
    else:
        y_true_encoded = y_true.astype(int)

    # Load Scaler
    scaler = None
    try:
        scaler = joblib.load('/app/feature_scaler.joblib')
    except Exception:
        pass

    features_list = []
    valid_idx = []
    for i, (name, p) in enumerate(zip(audio_names, audio_paths)):
        if not os.path.exists(p):
            warnings.warn(f"Audio not found: {p}. Skipping.")
            continue
        y = load_audio_full(p, target_sr=args.sr)
        feat = compute_features(y, sr=args.sr, n_mfcc=args.n_mfcc, fixed_len=args.fixed_len, scaler=scaler)
        # feat shape: (T, F)
        features_list.append(feat)
        valid_idx.append(i)

    if len(features_list) == 0:
        raise RuntimeError("No valid audio files to process.")

    # Run Inference
    probs = []
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"]) 
    input_name = sess.get_inputs()[0].name
    
    for feat in features_list:
        # Model expects (batch, time_steps, n_mfcc)
        x = feat[None, ...]  # (1, T, F)
        out = sess.run(None, {input_name: x})[0]
        
        # Handle various output shapes
        if out.ndim == 2:
            logit = out[0]
        elif out.ndim == 4:
            logit = out.mean(axis=(2,3))[0]  
        else:
            logit = out.reshape(out.shape[0], -1)[0]
        
        prob = softmax(logit)
        probs.append(prob)

    probs = np.stack(probs, axis=0)
    y_true_valid = y_true[valid_idx]
    y_true_valid_encoded = y_true_encoded[valid_idx]

    y_pred, metrics = evaluate_predictions(y_true_valid_encoded, probs)

    # Build Output DataFrame
    pred_labels = y_pred
    pred_rows = []
    for idx, vi in enumerate(valid_idx):
        gt_enc = int(y_true_valid_encoded[idx])
        pred_enc = int(pred_labels[idx])
        
        pred_rows.append({
            'audio_name': audio_names[vi],
            'prob_score': str(probs[idx]), # Full array as string
            'ground_truth': gt_enc,        # Encoded Integer
            'predicted': pred_enc,         # Encoded Integer
        })

    pred_df = pd.DataFrame(pred_rows)
    metrics_df = pd.DataFrame([metrics])
    
    # Class Mapping Sheet
    if le is not None:
        mapping_df = pd.DataFrame({'class_index': list(range(len(le.classes_))), 'class_label': le.classes_})
    else:
        mapping_df = pd.DataFrame()

    xlsx_path = out_dir / 'test_results.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Predictions', index=False)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        if not mapping_df.empty:
            mapping_df.to_excel(writer, sheet_name='ClassMapping', index=False)

    # Print metrics JSON for orchestration
    print(json.dumps(metrics))

if __name__ == '__main__':
    main()