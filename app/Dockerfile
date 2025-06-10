# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container at /app
# Note: 'app/' refers to the 'app' directory relative to the BUILD CONTEXT (which will now be the project root)
COPY app/app.py /app/app.py
COPY requirements.txt /app/requirements.txt # Make sure this is still correct if requirements.txt was in 'app/'
# Copy the trained model from the model/ directory (relative to build context) into /app
COPY model/fraud_model.pkl /app/fraud_model.pkl

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run app.py using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
