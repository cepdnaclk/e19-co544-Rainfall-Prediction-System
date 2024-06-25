FROM python:3.10.12

WORKDIR /code

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the image
COPY app/ /code/app/

# Copy model files into the app directory
COPY models/model_vavuniya.joblib /code/app/
COPY models/model_anuradhapura.joblib /code/app/
COPY models/model_maha.joblib /code/app/

# Expose port 8000 (assuming your application listens on this port)
EXPOSE 8000

# Set environment variable
ENV NAME RainfallPrediction

# Command to run your application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
