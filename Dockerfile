FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, and data
COPY . .

# Expose standard Gradio port
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
