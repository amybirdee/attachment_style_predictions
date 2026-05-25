from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.config import ARTIFACT_PATH, INPUT_TABLE, OUTPUT_TABLE, SCHEMA
from src.predict import predict_attachment_styles_to_sql


with DAG(
    dag_id = "attachment_style_batch_prediction",
    start_date = datetime(2026, 7, 1),
    schedule = None,
    catchup = False,
    tags = ["ml", "attachment-style", "batch-scoring"],
) as dag:

    run_predictions = PythonOperator(
        task_id = "run_attachment_style_predictions",
        python_callable = predict_attachment_styles_to_sql,
        op_kwargs = {
            "input_table": INPUT_TABLE,
            "output_table": OUTPUT_TABLE,
            "artifact_path": ARTIFACT_PATH,
            "schema": SCHEMA,
            "if_exists": "append",
        },
    )
