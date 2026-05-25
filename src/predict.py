import pickle
import pandas as pd

from src.config import ID_COLUMN, MODEL_VERSION
from src.database import get_db_engine
from src.preprocessing import prepare_features


def load_artifacts(artifact_path: str) -> dict:
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def predict_attachment_styles_to_sql(
    input_table: str,
    output_table: str,
    artifact_path: str,
    schema: str = "public",
    if_exists: str = "append",
) -> None:
    engine = get_db_engine()
    artifacts = load_artifacts(artifact_path)

    input_query = f"SELECT * FROM {schema}.{input_table};"
    df = pd.read_sql(input_query, engine)

    if ID_COLUMN not in df.columns:
        raise ValueError(f"Expected unique ID column '{ID_COLUMN}' in input table.")

    features = prepare_features(df, artifacts)

    model = artifacts["model"]

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    inverse_label_map = {v: k for k, v in artifacts["label_map"].items()}

    output = pd.DataFrame()
    output[ID_COLUMN] = df[ID_COLUMN]

    output["prediction_timestamp"] = pd.Timestamp.utcnow()
    output["model_version"] = MODEL_VERSION

    output["predicted_attachment_style_id"] = predictions
    output["predicted_attachment_style"] = [inverse_label_map[pred] for pred in predictions]

    for class_id, class_name in inverse_label_map.items():
        clean_class_name = class_name.lower().replace(" ", "_")
        output[f"probability_{clean_class_name}"] = probabilities[:, class_id]

    output.to_sql(
        name = output_table,
        con = engine,
        schema = schema,
        if_exists = if_exists,
        index = False,
    )
