FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/vector_db data/cache/embeddings data/cache/tokens

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "contextllm.main", "server", "--host", "0.0.0.0", "--port", "8000"]
