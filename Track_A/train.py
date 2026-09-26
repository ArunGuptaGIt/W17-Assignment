import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

# Set MLflow experiment
MLFLOW_EXPERIMENT_NAME = "telco_churn_experiment"
MODEL_REGISTRY_NAME = "telco_churn_model"

def load_and_preprocess_data(data_path: str):
    """Load and clean Telco Customer Churn dataset."""
    df = pd.read_csv(data_path)
    
    # Drop customerID as it is non-predictive
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Coerce TotalCharges to float, fill NaNs with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Map target variable Churn ('Yes'/'No') -> (1/0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    return X, y

def build_preprocessor(X: pd.DataFrame):
    """Create ColumnTransformer for preprocessing numeric and categorical features."""
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    
    return preprocessor

def evaluate_and_plot(model, X_test, y_test, run_name: str, artifact_dir: str):
    """Compute metrics and save confusion matrix & ROC curve plots."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
        
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    docs_img_dir = Path(__file__).parent.resolve() / "docs" / "images"
    docs_img_dir.mkdir(exist_ok=True, parents=True)
    clean_name = run_name.replace(" ", "_").lower()
    
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f"Confusion Matrix - {run_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    
    # Save permanent tracked copy for Git docs
    perm_cm_path = docs_img_dir / f"confusion_matrix_{clean_name}.png"
    plt.savefig(str(perm_cm_path), bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {run_name}")
    plt.legend(loc="lower right")
    roc_path = os.path.join(artifact_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches='tight')
    
    # Save permanent tracked copy for Git docs
    perm_roc_path = docs_img_dir / f"roc_curve_{clean_name}.png"
    plt.savefig(str(perm_roc_path), bbox_inches='tight')
    plt.close()
    
    return metrics, cm_path, roc_path

def main():
    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / "dataset" / "telco_churn.csv"
    if not data_path.exists():
        data_path = script_dir / "telco_churn.csv"
        
    print(f"Loading data from {data_path}...")
    X, y = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Model configs to run
    models_to_train = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            "params": {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"}
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=5, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 5}
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='logloss'),
            "params": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8}
        }
    ]
    
    run_results = []
    
    for config in models_to_train:
        run_name = config["name"]
        print(f"\n--- Training {run_name} ---")
        
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config["model"])
        ])
        
        with mlflow.start_run(run_name=run_name) as run:
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            artifact_dir = f"/tmp/artifacts_{run.info.run_id}"
            metrics, cm_path, roc_path = evaluate_and_plot(pipeline, X_test, y_test, run_name, artifact_dir)
            
            # Log params & metrics
            mlflow.log_params(config["params"])
            mlflow.log_param("model_family", run_name)
            mlflow.log_metrics(metrics)
            
            # Log artifacts
            mlflow.log_artifact(cm_path, artifact_path="plots")
            mlflow.log_artifact(roc_path, artifact_path="plots")
            
            # Log model with cloudpickle serialization format for tree compatibility
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=X_test.iloc[:2],
                serialization_format="cloudpickle"
            )
            
            run_results.append({
                "run_id": run.info.run_id,
                "model_name": run_name,
                "metrics": metrics,
                "pipeline": pipeline
            })
            
            print(f"Results for {run_name}:")
            for m, val in metrics.items():
                print(f"  {m}: {val:.4f}")
                
    # Model comparison
    print("\n==================================================")
    print("           MLflow Run Comparison Table")
    print("==================================================")
    headers = ["Model Name", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    print(f"{headers[0]:<20} | {headers[1]:<8} | {headers[2]:<9} | {headers[3]:<7} | {headers[4]:<8} | {headers[5]:<8}")
    print("-" * 75)
    
    best_run = None
    best_f1 = -1.0
    
    for r in run_results:
        m = r["metrics"]
        print(f"{r['model_name']:<20} | {m['accuracy']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['f1']:<8.4f} | {m['roc_auc']:<8.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_run = r
            
    print("-" * 75)
    print(f"\nWinning Model: {best_run['model_name']} with F1 Score = {best_f1:.4f} & ROC-AUC = {best_run['metrics']['roc_auc']:.4f}")
    
    # Model Registration & Stage Transition
    print(f"\nRegistering winning model ({best_run['model_name']}) in MLflow Model Registry...")
    model_uri = f"runs:/{best_run['run_id']}/model"
    registered_model = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    
    client = mlflow.tracking.MlflowClient()
    
    # Transition stage to Staging then Production
    try:
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered_model.version,
            stage="Staging"
        )
        print(f"Model version {registered_model.version} transitioned to Staging.")
        
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered_model.version,
            stage="Production"
        )
        print(f"Model version {registered_model.version} transitioned to Production.")
    except Exception as e:
        # MLflow 3.x alias transition fallback if stages are deprecated
        print(f"Stage transition fallback note: {e}")
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, "Staging", registered_model.version)
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, "Production", registered_model.version)
        print(f"Model version {registered_model.version} assigned alias 'Production'.")

if __name__ == "__main__":
    main()
