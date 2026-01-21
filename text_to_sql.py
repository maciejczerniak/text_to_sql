import json
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sqlalchemy import create_engine, inspect, text
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
except ImportError:
    px = None

SQL_TEMPLATE = """
You are a SQL expert. Given the following schema: {schema_details}, translate the user's
natural language question into a valid MySQL query. Return ONLY the SQL code
inside a code block.

User question: {query}
"""

FIX_TEMPLATE = """
You are a SQL expert. The following SQL failed to execute.
Schema: {schema_details}
User question: {query}
SQL:
```sql
{sql}
```
Error: {error}
Fix the query for MySQL. Return ONLY the corrected SQL code inside a code block.
"""

ANSWER_TEMPLATE = """
You are a data assistant. Given the user question and SQL results (JSON array of
objects), answer concisely. If results are empty, say that no data was found.
Do not mention SQL or the database.

Question: {query}
Results: {results}
Answer:
"""

ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
DISALLOWED = re.compile(
    r";|--|/\*|\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|grant|revoke)\b"
    r"|\binto\s+outfile\b|\binto\s+dumpfile\b|\bload_file\b|\bsleep\b|\bbenchmark\b",
    re.IGNORECASE,
)
CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TABLE_REF_RE = re.compile(r"\b(from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)", re.IGNORECASE)
DISALLOWED_SCHEMA_PREFIXES = {"information_schema", "mysql", "performance_schema", "sys", "pg_catalog"}


def load_dotenv_file(path: str = ".env", override: bool = False) -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                key, sep, value = line.partition("=")
                if not sep:
                    continue
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                if not override and key in os.environ:
                    continue
                os.environ[key] = value
    except FileNotFoundError:
        pass


load_dotenv_file(os.getenv("DOTENV_PATH", ".env"))


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:
            return st.secrets.get(key)
    except Exception:
        pass
    return os.getenv(key, default)


def build_db_url() -> Optional[str]:
    db_url = get_setting("DB_URL")
    if db_url:
        return db_url

    dialect = get_setting("DB_DIALECT", "sqlite")
    if dialect.startswith("sqlite"):
        return get_setting("SQLITE_PATH", "sqlite:///testdb.sqlite")

    host = get_setting("DB_HOST")
    user = get_setting("DB_USER")
    password = get_setting("DB_PASSWORD", "")
    name = get_setting("DB_NAME")
    port = get_setting("DB_PORT", "")

    if not host or not user or not name:
        return None

    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password) if password else ""
    auth = f"{user_enc}:{pass_enc}@" if password else f"{user_enc}@"
    hostport = f"{host}:{port}" if port else host
    return f"{dialect}://{auth}{hostport}/{name}"


def mask_db_url(url: str) -> str:
    return re.sub(r":([^:@/]+)@", ":****@", url)


def clean_text(text_value: str) -> str:
    cleaned_text = re.sub(r"<think>.*?</think>", "", text_value, flags=re.DOTALL)
    return cleaned_text.strip()


def strip_code_fences(text_value: str) -> str:
    if text_value.strip().startswith("```"):
        text_value = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text_value.strip(), flags=re.DOTALL)
    return text_value.strip()


def clean_sql(text_value: str) -> str:
    cleaned = clean_text(text_value)
    match = CODE_BLOCK_RE.search(cleaned)
    if match:
        cleaned = match.group(1)
    else:
        cleaned = strip_code_fences(cleaned)
    cleaned = re.sub(r"^\s*sql\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().rstrip(";")


def normalize_identifier(name: str) -> str:
    return name.strip().strip('`"[]')


def extract_table_refs(sql: str) -> List[str]:
    return [match[1] for match in TABLE_REF_RE.findall(sql)]


def has_disallowed_prefix(ref: str) -> bool:
    parts = [normalize_identifier(part).lower() for part in ref.split(".") if part]
    for prefix in parts[:-1]:
        if prefix in DISALLOWED_SCHEMA_PREFIXES:
            return True
    return False


def normalize_table_name(ref: str) -> str:
    parts = [normalize_identifier(part) for part in ref.split(".") if part]
    if not parts:
        return ""
    return parts[-1]


def is_safe_sql(sql: str, schema: Dict[str, List[str]]) -> bool:
    if not ALLOWED_START.match(sql):
        return False
    if DISALLOWED.search(sql):
        return False
    refs = extract_table_refs(sql)
    for ref in refs:
        if has_disallowed_prefix(ref):
            return False
    tables = {normalize_table_name(ref) for ref in refs if normalize_table_name(ref)}
    if not tables:
        return True
    allowed = {name.lower() for name in schema.keys()}
    return all(table.lower() in allowed for table in tables)


def ensure_limit(sql: str, limit: int = 200) -> str:
    if re.search(r"\blimit\b|\bfetch\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {limit}"


@st.cache_resource(show_spinner=False)
def get_engine(db_url: str):
    return create_engine(db_url, pool_pre_ping=True)


@st.cache_data(show_spinner=False)
def extract_schema(db_url: str) -> Dict[str, List[str]]:
    engine = get_engine(db_url)
    inspector = inspect(engine)
    schema: Dict[str, List[str]] = {}

    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        schema[table] = [col["name"] for col in columns]

    return schema


def to_sql_query(query: str, schema_details: str, model: OllamaLLM) -> str:
    prompt = ChatPromptTemplate.from_template(SQL_TEMPLATE)
    chain = prompt | model
    return clean_sql(chain.invoke({"query": query, "schema_details": schema_details}))


def fix_sql_query(query: str, schema_details: str, sql: str, error: str, model: OllamaLLM) -> str:
    prompt = ChatPromptTemplate.from_template(FIX_TEMPLATE)
    chain = prompt | model
    return clean_sql(
        chain.invoke({"query": query, "schema_details": schema_details, "sql": sql, "error": error})
    )


def generate_answer(query: str, rows: List[Dict[str, Any]], model: OllamaLLM) -> str:
    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    chain = prompt | model
    results_json = json.dumps(rows, ensure_ascii=True)
    return clean_text(chain.invoke({"query": query, "results": results_json}))


def run_query(db_url: str, sql: str) -> List[Dict[str, Any]]:
    engine = get_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]


def query_with_retries(
    query: str,
    schema_details: str,
    schema: Dict[str, List[str]],
    db_url: str,
    model: OllamaLLM,
    limit: int,
    max_retries: int,
) -> Dict[str, Any]:
    sql = to_sql_query(query, schema_details, model)
    seen_sql = set()
    last_error = None

    for attempt in range(max_retries + 1):
        sql = ensure_limit(sql, limit=limit)
        if not is_safe_sql(sql, schema):
            return {"sql": sql, "rows": None, "error": "Generated SQL was not a safe SELECT statement."}
        if sql in seen_sql:
            return {"sql": sql, "rows": None, "error": "Generated SQL repeated without fixing the error."}
        seen_sql.add(sql)
        try:
            rows = run_query(db_url, sql)
            return {"sql": sql, "rows": rows, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            sql = fix_sql_query(query, schema_details, sql, last_error, model)

    return {"sql": sql, "rows": None, "error": f"Query failed: {last_error}"}


def coerce_datetime_columns(df):
    if pd is None:
        return df
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
    return df


def build_chart(df):
    if pd is None or px is None or df.empty:
        return None
    df = coerce_datetime_columns(df.copy())
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    x_col = None
    for col in df.columns:
        if col not in numeric_cols:
            x_col = col
            break
    if x_col is None:
        df = df.reset_index(drop=True)
        df["index"] = df.index
        x_col = "index"

    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        return px.line(df, x=x_col, y=numeric_cols)
    y_cols = numeric_cols if len(numeric_cols) > 1 else numeric_cols[0]
    return px.bar(df, x=x_col, y=y_cols)


def render_results(rows: List[Dict[str, Any]]) -> None:
    if rows:
        if pd is None:
            st.dataframe(rows, use_container_width=True)
            return
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        fig = build_chart(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rows returned.")


st.set_page_config(page_title="Text-to-SQL Chat")

OLLAMA_MODEL = get_setting("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL") or get_setting("OLLAMA_HOST")

if OLLAMA_BASE_URL:
    model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
else:
    model = OllamaLLM(model=OLLAMA_MODEL)

QUERY_LIMIT = int(get_setting("QUERY_LIMIT", "200"))
SQL_MAX_RETRIES = int(get_setting("SQL_MAX_RETRIES", "2"))

db_url = build_db_url()
if not db_url:
    st.error(
        "Database settings are missing. Set DB_URL or DB_HOST/DB_USER/DB_PASSWORD/DB_NAME."
    )
    st.stop()

st.title("Text-to-SQL Chat")
st.caption(f"Model: {OLLAMA_MODEL} | Database: {mask_db_url(db_url)}")
show_work = st.toggle("Show your work", value=False, key="show_work")
if pd is None or px is None:
    st.caption("Charts disabled until pandas and plotly are installed.")

try:
    schema = extract_schema(db_url)
    schema_details = json.dumps(schema, ensure_ascii=True)
except Exception as exc:
    st.error(f"Could not load schema: {exc}")
    st.stop()

with st.expander("Database schema"):
    st.code(schema_details, language="json")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("rows") is not None:
                render_results(message["rows"])
            if message.get("error"):
                st.error(message["error"])
            if show_work and message.get("sql"):
                st.code(message["sql"], wrap_lines=True, language="sql")

user_prompt = st.chat_input("Ask about your data")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            result = query_with_retries(
                user_prompt,
                schema_details,
                schema,
                db_url,
                model,
                QUERY_LIMIT,
                SQL_MAX_RETRIES,
            )

        if result["error"]:
            error = result["error"]
            st.error(error)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I ran into an error while answering that.",
                    "error": error,
                    "sql": result.get("sql"),
                }
            )
            st.stop()

        sql = result["sql"]
        rows = result["rows"]

        answer_rows = rows[: int(get_setting("ANSWER_MAX_ROWS", "50"))]
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(user_prompt, answer_rows, model)
            except Exception as exc:
                answer = f"I could not generate an answer: {exc}"
        st.markdown(answer)
        render_results(rows)
        if show_work:
            st.code(sql, wrap_lines=True, language="sql")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sql": sql,
                "rows": rows,
            }
        )
