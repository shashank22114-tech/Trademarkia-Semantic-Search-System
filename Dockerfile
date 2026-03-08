# Use an official, lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install the lightweight CPU-only version of PyTorch first to prevent timeouts
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]