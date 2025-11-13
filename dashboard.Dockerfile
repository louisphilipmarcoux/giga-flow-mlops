FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY src/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src/dashboard/app.py .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]