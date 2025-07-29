# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Prophet & NeuralProphet require them)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    libfreetype6-dev \
    libpng-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all application code
COPY . .

# Expose port (used by Cloud Run / Render)
EXPOSE 8080

# Default run command (use uvicorn in production mode)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
