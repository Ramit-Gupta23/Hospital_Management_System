# ============================================================
#  Dockerfile — Hospital LOS Prediction API
#  Build : docker build -t hospital-los-api .
#  Run   : docker run -p 8000:8000 hospital-los-api
#  Docs  : http://localhost:8000/docs
# ============================================================

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create folders the app needs at runtime
RUN mkdir -p logs models reports

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the server
# --host 0.0.0.0 makes it reachable outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
