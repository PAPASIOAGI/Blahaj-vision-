# Base image dengan Python dan system dependencies untuk OpenCV
FROM python:3.11-slim

# Install system dependencies untuk OpenCV (libGL, libglib, dll)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first untuk cache layer
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory (weights akan di-download saat runtime atau di-mount)
RUN mkdir -p models

# Expose port jika diperlukan (untuk web streaming nanti)
# EXPOSE 8000

# Default command
CMD ["python", "src/main.py"]