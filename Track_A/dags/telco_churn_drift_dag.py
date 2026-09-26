from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import os

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_evidently_drift_check():
    """Execute Track A Evidently AI drift monitoring script."""
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'drift_monitoring.py')
    print(f"Executing drift check script at {script_path}...")
    res = subprocess.run(['uv', 'run', 'python', script_path], capture_output=True, text=True, check=True)
    print("Drift check output:")
    print(res.stdout)
    if "Dataset Drift Detected: True" in res.stdout or "Drifted Columns Count" in res.stdout:
        print("[ALERT] Significant data/target drift detected! Alerting model governance and triggering retraining recommendation.")

with DAG(
    'telco_churn_drift_monitoring_dag',
    default_args=default_args,
    description='Scheduled Evidently AI Data Drift & Target Drift monitoring for Telco Churn pipeline',
    schedule_interval='0 0 * * *',  # Daily at midnight
    catchup=False,
) as dag:

    drift_check_task = PythonOperator(
        task_id='run_drift_and_target_monitoring',
        python_callable=run_evidently_drift_check,
    )
