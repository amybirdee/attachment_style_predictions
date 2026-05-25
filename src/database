import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config import DOTENV_PATH


def get_db_engine():
    load_dotenv(dotenv_path = DOTENV_PATH)

    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    return create_engine(db_url)
