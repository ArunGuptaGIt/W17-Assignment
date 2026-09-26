import os
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from dataclasses import dataclass

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.metrics import ColumnSummaryMetric

def calculate_custom_segment_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, column_name: str = "MonthlyCharges", segment_column: str = "Contract", segment_value: str = "Month-to-month"):
    """Custom metric function calculating mean difference of a numeric column for a specific segment."""
    ref_segment = reference_data[reference_data[segment_column] == segment_value]
    curr_segment = current_data[current_data[segment_column] == segment_value]

    ref_mean = float(ref_segment[column_name].mean())
    curr_mean = float(curr_segment[column_name].mean())
    diff = curr_mean - ref_mean

    return {
        "segment_name": f"{segment_column}=='{segment_value}'",
        "column_name": column_name,
        "reference_mean": ref_mean,
        "current_mean": curr_mean,
        "mean_difference": diff
    }

def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df

def inject_synthetic_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Inject deliberate synthetic drift into current dataset."""
    df_drifted = df.copy()
    np.random.seed(42)

    # 1. Numeric Drift: Scale and shift MonthlyCharges & tenure
    df_drifted['MonthlyCharges'] = df_drifted['MonthlyCharges'] * 1.35 + np.random.normal(15, 5, size=len(df_drifted))
    df_drifted['tenure'] = df_drifted['tenure'] * 0.6 + np.random.normal(-5, 2, size=len(df_drifted))
    df_drifted['tenure'] = df_drifted['tenure'].clip(lower=0)

    # 2. Categorical Drift: Skew Contract distribution to 85% Month-to-month
    mask = np.random.rand(len(df_drifted)) < 0.85
    df_drifted.loc[mask, 'Contract'] = 'Month-to-month'

    # 3. Target Drift: Flip 20% of Churn labels
    churn_mask = np.random.rand(len(df_drifted)) < 0.20
    df_drifted.loc[churn_mask, 'Churn'] = df_drifted.loc[churn_mask, 'Churn'].map({'Yes': 'No', 'No': 'Yes'})

    return df_drifted

def main():
    print("==================================================")
    print("     Track A: Evidently AI Drift Monitoring")
    print("==================================================")

    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / "dataset" / "telco_churn.csv"
    if not data_path.exists():
        data_path = script_dir / "telco_churn.csv"

    df = load_data(data_path)

    # Split dataset: 70% Reference, 30% Current
    train_size = int(0.7 * len(df))
    reference_data = df.iloc[:train_size].copy()
    current_data = df.iloc[train_size:].copy()

    print(f"Reference dataset size: {len(reference_data)} rows")
    print(f"Current raw dataset size: {len(current_data)} rows")

    # Inject synthetic drift into current data
    print("Injecting synthetic drift into current dataset...")
    current_data_drifted = inject_synthetic_drift(current_data)

    reports_dir = script_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 1. Run Data Drift Report (with ColumnSummaryMetric included)
    print("\n--- Generating Data Drift Report ---")
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        ColumnSummaryMetric(column_name="MonthlyCharges")
    ])
    data_drift_report.run(reference_data=reference_data, current_data=current_data_drifted)
    
    data_drift_html_path = reports_dir / "data_drift_report.html"
    data_drift_report.save_html(str(data_drift_html_path))
    print(f"Data Drift Report saved to {data_drift_html_path}")

    # 2. Run Target Drift Report
    print("\n--- Generating Target Drift Report ---")
    target_drift_report = Report(metrics=[
        TargetDriftPreset()
    ])
    target_drift_report.run(reference_data=reference_data, current_data=current_data_drifted)
    
    target_drift_html_path = reports_dir / "target_drift_report.html"
    target_drift_report.save_html(str(target_drift_html_path))
    print(f"Target Drift Report saved to {target_drift_html_path}")

    # 3. Calculate Custom Segment Metric
    custom_metric_res = calculate_custom_segment_drift(reference_data, current_data_drifted)

    # Extract drift metrics summary
    drift_res = data_drift_report.as_dict()
    print("\n--------------------------------------------------")
    print("               Drift Summary Analysis")
    print("--------------------------------------------------")
    try:
        metrics_list = drift_res.get("metrics", [])
        for m in metrics_list:
            res = m.get("result", {})
            if "dataset_drift" in res:
                print(f"Dataset Drift Detected: {res['dataset_drift']}")
                print(f"Drifted Columns Count: {res.get('number_of_drifted_columns', 0)} / {res.get('number_of_columns', 0)}")
                drifted_cols = [
                    col for col, cdict in res.get('drift_by_columns', {}).items()
                    if cdict.get('drift_detected', False)
                ]
                print(f"Drifted Feature List: {drifted_cols}")
    except Exception as e:
        print(f"Parsing metric summary error: {e}")

    print(f"\nCustom Segment Metric ({custom_metric_res['segment_name']}):")
    print(f"  Reference Mean {custom_metric_res['column_name']}: ${custom_metric_res['reference_mean']:.2f}")
    print(f"  Current Mean {custom_metric_res['column_name']}: ${custom_metric_res['current_mean']:.2f}")
    print(f"  Mean Shift: ${custom_metric_res['mean_difference']:+.2f}")

    # Log to MLflow
    mlflow.set_experiment("telco_churn_monitoring")
    with mlflow.start_run(run_name="evidently_drift_analysis") as run:
        mlflow.log_param("reference_rows", len(reference_data))
        mlflow.log_param("current_rows", len(current_data_drifted))
        mlflow.log_param("synthetic_drift_injected", True)

        mlflow.log_metric("custom_monthly_charges_ref_mean", custom_metric_res["reference_mean"])
        mlflow.log_metric("custom_monthly_charges_curr_mean", custom_metric_res["current_mean"])
        mlflow.log_metric("custom_monthly_charges_mean_shift", custom_metric_res["mean_difference"])

        # Log reports as MLflow artifacts
        mlflow.log_artifact(str(data_drift_html_path), artifact_path="evidently_reports")
        mlflow.log_artifact(str(target_drift_html_path), artifact_path="evidently_reports")
        print(f"\nEvidently reports successfully logged to MLflow run '{run.info.run_id}'.")

if __name__ == "__main__":
    main()
