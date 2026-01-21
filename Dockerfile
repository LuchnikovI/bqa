# BQA - Quantum Annealing Simulator
# Docker image for running examples (installs from Test PyPI)
# TODO: Change to production PyPI after official release

FROM python:3.12-slim

WORKDIR /app

# Install bqa from Test PyPI (with fallback to production PyPI for dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bqa

# Copy examples directory
COPY examples/ /app/examples/

# Default command: run the full-size IBM heavy hex example
CMD ["python", "examples/full_size_ibm_heavy_hex.py"]
