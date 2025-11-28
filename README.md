# So Hum Challenge: Audio Classification

## Overview
This project provides a Dockerized solution for classifying cardiopulmonary audio into Heart Sounds, Lung Sounds, and Noise. It uses a hybrid CNN-LSTM model trained on MFCC features.

## Files Structure
* `Dockerfile`: Configuration to build the self-contained environment.
* `inferencing.py`: Main script for loading data, preprocessing, and generating predictions.
* `model.onnx`: The trained model weights in standard ONNX format.
* `feature_scaler.joblib`: The pre-fitted scaler for data normalization.
* `requirements.txt`: Python dependencies.

## Methodology
* **Input:** Raw .wav audio files (resampled to 16kHz).
* **Features:** 13 MFCCs.
* **Architecture:** 1D CNN (Feature Extractor) -> LSTM (Temporal Modeler) -> Dense (Classifier).
* **Inference:** Supports full-audio level prediction on variable length inputs.

## How to Run (Docker)

1. **Build the Image:**
   docker build -t filename:latest .
2. **Save the image:**
   docker save -o filename.tar filename:latest
4. **Load the image:**
   docker load -i filename.tar
6.  **Run Inference:** 
   * Ensure your test_input folder contains raw_audio and test_meta_data.csv.

     docker run --rm -v "%cd%\test_input:/test_input" -v "%cd%\output:/output" filename:latest

7. **Output:**
     Results are saved to /output/test_results.xlsx containing Predictions, Metrics, and Class Mappings.
