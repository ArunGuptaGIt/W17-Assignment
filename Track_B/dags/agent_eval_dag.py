from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import os

default_args = {
    'owner': 'mlops_agentic_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_agentic_regression_eval():
    """Execute Track B prompt experiment & regression evaluation harness."""
    eval_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval', 'run_prompt_experiments.py')
    print(f"Executing agentic assistant evaluation harness at {eval_script}...")
    res = subprocess.run(['uv', 'run', 'python', eval_script], capture_output=True, text=True, check=True)
    print("Agentic Eval Output:")
    print(res.stdout)
    
    if "100.0" not in res.stdout and "prompt_v3" in res.stdout:
        print("[ALERT] Regression detected! Agent pass rate degraded below target threshold (100%). Block deployment!")

with DAG(
    'agentic_assistant_regression_eval_dag',
    default_args=default_args,
    description='Scheduled regression evaluation harness & MLflow trace logging for Track B Agentic AI Assistant',
    schedule_interval='0 2 * * *',  # Daily at 2:00 AM
    catchup=False,
) as dag:

    agent_eval_task = PythonOperator(
        task_id='run_agent_eval_and_regression_checks',
        python_callable=run_agentic_regression_eval,
    )
