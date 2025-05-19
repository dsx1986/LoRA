FROM debian:bookworm

WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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

# Add virtual environment activation to .bashrc
RUN echo 'source /opt/venv/bin/activate' >> /root/.bashrc

# Set the default command
CMD ["python", "-c", "import loralib; print('LoRA library is installed and ready to use!')"]
