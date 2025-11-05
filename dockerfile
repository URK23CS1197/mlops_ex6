FROM python:3.8-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all other source files
COPY . .

# Expose correct port your app runs on (usually Flask defaults to 5000)
EXPOSE 5000

# Use Gunicorn for production-grade server instead of 'python app.py'
# Replace 'app:app' with your actual Flask app file and app variable names
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
