# Use a lightweight Python base image suitable for production
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies first.
# This leverages Docker's layer caching for faster builds if dependencies don't change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application code.
# The trained model ('fraud_model.pkl') will be copied into this directory by Jenkins
# BEFORE the Docker build is run in the Jenkins build step.
COPY app.py .

# Expose the port on which the Flask application will run
EXPOSE 5000

# Command to run the application using Gunicorn (a production-grade WSGI server)
# 'app:app' refers to the 'app' Flask instance within the 'app.py' module
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
