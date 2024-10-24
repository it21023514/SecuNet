FROM python:3.12.7-slim

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Create a virtual environment
RUN python -m venv $VIRTUAL_ENV

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the default port for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "FrontEnd.py"]
