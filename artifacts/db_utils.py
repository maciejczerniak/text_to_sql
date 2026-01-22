from typing import Any, Dict, List

import streamlit as st
from sqlalchemy import create_engine, inspect, text


# Cache the database engine between reruns.
@st.cache_resource(show_spinner=False)
def get_engine(db_url: str):
    """Create and cache a SQLAlchemy engine."""
    return create_engine(db_url, pool_pre_ping=True)


# Cache schema discovery to reduce DB roundtrips.
@st.cache_data(show_spinner=False)
def extract_schema(db_url: str) -> Dict[str, List[str]]:
    """Inspect tables/columns and return a schema dictionary."""
    engine = get_engine(db_url)
    inspector = inspect(engine)
    schema: Dict[str, List[str]] = {}

    # Capture table names and their columns for the prompt and safety checks.
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        schema[table] = [col["name"] for col in columns]

    return schema


# Execute SQL and return rows as dictionaries.
def run_query(db_url: str, sql: str) -> List[Dict[str, Any]]:
    """Execute SQL against the DB and return row dicts."""
    engine = get_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]
