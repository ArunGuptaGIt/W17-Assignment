import time
import requests
import subprocess
import sys
import json
from pathlib import Path

def test_serving():
    print("Testing Track A FastAPI Model Serving...")
    
    # 1. Sample requests
    sample_single = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    sample_batch = {
        "inputs": [
            sample_single,
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 45,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 105.5,
                "TotalCharges": 4747.5
            }
        ]
    }
    
    # Direct testing via FastAPI TestClient / requests to running process
    from fastapi.testclient import TestClient
    from serve import app
    
    with TestClient(app) as client:
        # Test Health endpoint
        response_health = client.get("/health")
        print("\n--- GET /health Response ---")
        print(f"Status Code: {response_health.status_code}")
        print(json.dumps(response_health.json(), indent=2))
        assert response_health.status_code == 200
        
        # Test Predict endpoint with single request
        response_single = client.post("/predict", json=sample_single)
        print("\n--- POST /predict (Single Customer) Response ---")
        print(f"Status Code: {response_single.status_code}")
        print(json.dumps(response_single.json(), indent=2))
        assert response_single.status_code == 200
        assert "predictions" in response_single.json()
        
        # Test Predict endpoint with batch request
        response_batch = client.post("/predict", json=sample_batch)
        print("\n--- POST /predict (Batch Customers) Response ---")
        print(f"Status Code: {response_batch.status_code}")
        print(json.dumps(response_batch.json(), indent=2))
        assert response_batch.status_code == 200
        assert len(response_batch.json()["predictions"]) == 2

    print("\nServing tests passed successfully!")

if __name__ == "__main__":
    test_serving()
