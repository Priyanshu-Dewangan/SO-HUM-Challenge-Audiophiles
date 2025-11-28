# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy model and inference code
COPY inferencing.py /app/inferencing.py
COPY model.onnx /app/model.onnx
COPY feature_scaler.joblib /app/feature_scaler.joblib

# Create mount points for organizer
VOLUME ["/test_input", "/output"]

# Default command: read from /test_input and write to /output
ENTRYPOINT ["python", "/app/inferencing.py", "--data", "/test_input", "--output", "/output"]
