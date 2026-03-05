# Backend Dockerfile — AI Disease Diagnosis Platform
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and PDF generation
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    xfonts-75dpi \
    xfonts-base \
    && curl -fsSL https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-3/wkhtmltox_0.12.6.1-3.bookworm_amd64.deb -o wkhtmltox.deb \
    && apt-get install -y ./wkhtmltox.deb \
    && rm wkhtmltox.deb \
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
