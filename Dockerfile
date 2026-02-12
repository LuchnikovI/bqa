# BQA - Quantum Annealing Simulator
# Docker image for running examples

FROM python:3.12-slim

WORKDIR /app

# Install pip + awscli (for S3 download inside Batch jobs)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir awscli

# Install bqa from official PyPI
RUN pip install --no-cache-dir bqa

# Copy examples directory
COPY examples/ /app/examples/

# Default command: run the full-size IBM heavy hex example
CMD ["python", "examples/full_size_ibm_heavy_hex.py"]
