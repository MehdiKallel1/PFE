# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port your Flask app runs on (default 5000)
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "run.py"]
