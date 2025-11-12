#!/usr/bin/env python3
"""
Debug script to test MLflow connectivity and model loading
Run this inside the model_service container to diagnose issues
"""

import os
import sys

print("=" * 60)
print("GIGAFLOW MODEL SERVICE DIAGNOSTICS")
print("=" * 60)

# 1. Check environment variables
print("\n1. ENVIRONMENT VARIABLES:")
print(f"   MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}")
print(f"   KAFKA_SERVER: {os.getenv('KAFKA_SERVER', 'NOT SET')}")
print(f"   DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT SET')[:50]}...")

# 2. Test MLflow server connectivity
print("\n2. TESTING MLFLOW SERVER CONNECTIVITY:")
try:
    import requests
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    response = requests.get(f"{mlflow_uri}/health", timeout=5)
    print(f"   ✓ MLflow server is reachable: {response.status_code}")
except Exception as e:
    print(f"   ✗ Cannot reach MLflow server: {e}")
    print("   This is likely the problem!")

# 3. Check if mlruns directory is mounted
print("\n3. CHECKING MLRUNS DIRECTORY:")
mlruns_path = "/mlruns"
if os.path.exists(mlruns_path):
    print(f"   ✓ {mlruns_path} exists")
    try:
        contents = os.listdir(mlruns_path)
        print(f"   Contents: {contents[:5]}...")  # Show first 5 items
    except Exception as e:
        print(f"   ✗ Cannot list directory: {e}")
else:
    print(f"   ✗ {mlruns_path} does not exist")
    print("   Volume mount may be incorrect!")

# 4. Try to import and set up MLflow
print("\n4. TESTING MLFLOW IMPORT AND SETUP:")
try:
    import mlflow
    print("   ✓ MLflow imported successfully")
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"   ✓ Tracking URI set to: {mlflow_uri}")
    
    # Try to list experiments
    client = mlflow.MlflowClient()
    experiments = client.search_experiments()
    print(f"   ✓ Found {len(experiments)} experiments")
    
except Exception as e:
    print(f"   ✗ MLflow setup failed: {e}")
    import traceback
    traceback.print_exc()

# 5. Try to load the model
print("\n5. TESTING MODEL LOADING:")
MLFLOW_MODEL_NAME = "giga-flow-sentiment"
MLFLOW_MODEL_ALIAS = "champion"

try:
    import mlflow.pyfunc
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    print(f"   Attempting to load: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    print("   ✓ Model loaded successfully!")
    
    # Try a test prediction
    import pandas as pd
    test_data = pd.DataFrame({'text': ['This is a test']})
    prediction = model.predict(test_data)
    print(f"   ✓ Test prediction successful: {prediction}")
    
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)