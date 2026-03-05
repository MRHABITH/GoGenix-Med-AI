# Backend Dockerfile — AI Disease Diagnosis Platform
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and PDF generation
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Install CPU-only PyTorch first to drastically reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
