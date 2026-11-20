FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies for espeak-ng.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

COPY .docker/run.py .

# Set the entrypoint
ENTRYPOINT ["python", "run.py"]
