import os
import uvicorn
import pandas as pd
from typing import List, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI(
    title="Telco Customer Churn Model Serving",
    description="FastAPI service serving production model loaded from MLflow Model Registry.",
    version="1.0.0"
)

MODEL_NAME = "telco_churn_model"
MODEL_STAGE = "Production"

# Load model on startup
model = None

@app.on_event("startup")
def load_mlflow_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading MLflow model from {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load from stage '{MODEL_STAGE}' ({e}). Attempting fallback to latest registered version...")
        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(MODEL_NAME)
            if latest_versions:
                latest_ver = latest_versions[-1].version
                fallback_uri = f"models:/{MODEL_NAME}/{latest_ver}"
                print(f"Loading fallback model from {fallback_uri}...")
                model = mlflow.pyfunc.load_model(fallback_uri)
                print("Fallback model loaded successfully.")
            else:
                raise RuntimeError("No registered model versions found.")
        except Exception as fallback_err:
            print(f"Failed to load model: {fallback_err}")

def get_model():
    global model
    if model is None:
        load_mlflow_model()
    return model

class SingleCustomerRequest(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

class BatchCustomerRequest(BaseModel):
    inputs: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    m = get_model()
    return {
        "status": "healthy",
        "model_loaded": m is not None,
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE
    }

@app.post("/predict")
def predict(payload: Union[SingleCustomerRequest, BatchCustomerRequest, Dict[str, Any]]):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Standardize input to DataFrame
        if isinstance(payload, SingleCustomerRequest):
            df_input = pd.DataFrame([payload.model_dump()])
        elif isinstance(payload, BatchCustomerRequest):
            df_input = pd.DataFrame(payload.inputs)
        elif isinstance(payload, dict):
            if "inputs" in payload:
                df_input = pd.DataFrame(payload["inputs"])
            else:
                df_input = pd.DataFrame([payload])
        else:
            raise HTTPException(status_code=400, detail="Invalid payload format.")

        preds = m.predict(df_input)
        
        # Determine probabilities if available
        probabilities = None
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(df_input)
            probabilities = probs.tolist()
        elif hasattr(m, "_model_impl") and hasattr(m._model_impl, "predict_proba"):
            probs = m._model_impl.predict_proba(df_input)
            probabilities = probs.tolist()
            
        results = []
        for idx, pred in enumerate(preds):
            churn_val = int(pred)
            res = {
                "churn_prediction": churn_val,
                "churn_label": "Yes" if churn_val == 1 else "No"
            }
            if probabilities is not None:
                res["probability_churn"] = round(float(probabilities[idx][1]), 4)
            results.append(res)
            
        return {
            "status": "success",
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
