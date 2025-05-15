FROM --platform=linux/amd64 python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-dev.txt ./
# Install PyTorch CPU version and other dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install pytest pytest-cov

# Copy the entire git repository including .git directory
COPY . .
# Explicitly copy the .git directory
COPY .git /app/.git

# Install the package in development mode
RUN pip install -e .

# Set the default command
CMD ["python", "-c", "import loralib; print('LoRA library is installed and ready to use!')"]
