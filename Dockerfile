# ---- Base image ----
FROM python:3.10-slim

# ---- Environment settings ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Set working directory ----
WORKDIR /app

# ---- System dependencies (CRITICAL: librdkafka-dev for confluent-kafka) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip ----
RUN pip install --upgrade pip

# ---- Copy requirements first (better caching) ----
COPY requirements.txt .

# ---- Install Python dependencies ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# ---- Expose Streamlit port (HF Spaces uses 7860) ----
EXPOSE 7860

# ---- Run Streamlit ----
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
