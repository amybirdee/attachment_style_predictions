import pandas as pd
from sentence_transformers import SentenceTransformer


def prepare_features(input_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = input_df.copy()

    text_col = artifacts["text_col"]
    id_col = artifacts["id_col"]

    if "parenting_style" in df.columns:
        df["parenting_style"] = df["parenting_style"].fillna("N/A")

    text_data = df[text_col].fillna("").astype(str)

    structured_df = df.drop(columns=[text_col, id_col], errors = "ignore")

    embedding_model = SentenceTransformer(artifacts["embedding_model_name"])

    text_embeddings = embedding_model.encode(
        text_data.tolist(),
        batch_size = 32,
        show_progress_bar = False,
    )

    embeddings_df = pd.DataFrame(
        text_embeddings,
        columns = artifacts["embedding_columns"],
        index = df.index,
    )

    structured_processed = artifacts["structured_preprocessor"].transform(structured_df)

    structured_processed_df = pd.DataFrame(
        structured_processed,
        columns = artifacts["structured_preprocessor"].get_feature_names_out(),
        index = df.index,
    )

    combined_df = pd.concat([structured_processed_df, embeddings_df], axis = 1)

    return combined_df[artifacts["feature_columns"]]
