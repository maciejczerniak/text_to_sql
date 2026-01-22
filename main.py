import json
import os
import re
from typing import Any, Dict, List
from urllib.parse import urlsplit

# Text-to-SQL pipeline:
# 1) Build schema context from the database.
# 2) Ask the model for SQL (and fix it on errors).
# 3) Execute the query, summarize the results, and render tables/charts.

import streamlit as st
from langchain_ollama.llms import OllamaLLM

from artifacts.config_utils import build_db_url, get_setting, load_dotenv_file
from artifacts.db_utils import extract_schema, run_query
from artifacts.llm_utils import fix_sql_query, generate_answer, infer_range_answer, to_sql_query
from artifacts.sql_safety import ensure_limit, is_safe_sql
from artifacts.viz_utils import charts_available, render_results

# Load environment variables from a local .env file when present.
load_dotenv_file(os.getenv("DOTENV_PATH", ".env"))


def wants_chart(question: str) -> bool:
    """Heuristic for chart requests in the user's question."""
    return bool(re.search(r"\b(plot|chart|graph|visuali[sz]e|trend|line|bar)\b", question, re.I))


def get_db_display_name(db_url: str) -> str:
    """Return a friendly database name for display."""
    name = get_setting("DB_NAME")
    if name:
        return name
    if db_url.startswith("sqlite"):
        path = db_url.split("///", 1)[-1]
        return os.path.basename(path) or "sqlite"
    parsed = urlsplit(db_url)
    if parsed.path:
        return parsed.path.lstrip("/") or "database"
    return "database"

# Run SQL with retry loops and model-based fixes.
def query_with_retries(
    query: str,
    schema_details: str,
    schema: Dict[str, List[str]],
    db_url: str,
    model: OllamaLLM,
    limit: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Generate SQL, validate it, execute it, and retry with fixes if needed."""
    sql = to_sql_query(query, schema_details, model)
    seen_sql = set()
    last_error = None

    # Retry loop for query generation and fix attempts.
    for attempt in range(max_retries + 1):
        # Enforce limits and safety checks before execution.
        sql = ensure_limit(sql, limit=limit)
        # Stop early if the SQL violates safety rules.
        if not is_safe_sql(sql, schema):
            return {"sql": sql, "rows": None, "error": "Generated SQL was not a safe SELECT statement."}
        # Guard against repeated SQL that never fixes the error.
        if sql in seen_sql:
            return {"sql": sql, "rows": None, "error": "Generated SQL repeated without fixing the error."}
        seen_sql.add(sql)
        try:
            rows = run_query(db_url, sql)
            return {"sql": sql, "rows": rows, "error": None}
        except Exception as exc:
            last_error = str(exc)
            # Stop after the final retry.
            if attempt >= max_retries:
                break
            # Feed the DB error back to the model to repair the query.
            sql = fix_sql_query(query, schema_details, sql, last_error, model)

    return {"sql": sql, "rows": None, "error": f"Query failed: {last_error}"}


st.set_page_config(page_title="Text-to-SQL Chat")

# Model and retry settings.
OLLAMA_MODEL = get_setting("OLLAMA_MODEL")
OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL") or get_setting("OLLAMA_HOST")

if not OLLAMA_MODEL:
    st.error("OLLAMA_MODEL is missing. Set it in .env or .streamlit/secrets.toml.")
    st.stop()

# Use a remote Ollama base URL when configured; otherwise default to local.
if OLLAMA_BASE_URL:
    model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
else:
    model = OllamaLLM(model=OLLAMA_MODEL)

QUERY_LIMIT = int(get_setting("QUERY_LIMIT", "200"))
SQL_MAX_RETRIES = int(get_setting("SQL_MAX_RETRIES", "2"))

# Resolve DB connection info.
db_url = build_db_url()
# Stop early if we cannot build a valid DB URL.
if not db_url:
    st.error(
        "Database settings are missing. Set DB_URL or DB_HOST/DB_USER/DB_PASSWORD/DB_NAME."
    )
    st.stop()

st.title("Text-to-SQL Chat")
db_display_name = get_db_display_name(db_url)
st.caption(f"Model: {OLLAMA_MODEL} | Database: {db_display_name}")
# "Show your work" reveals the SQL after the answer.
show_work = st.toggle(
    "Show your work",
    value=False,
    key="show_work",
    help="Show the generated SQL after each answer.",
)
show_chart = st.toggle(
    "Plot results",
    value=False,
    key="show_chart",
    help="Render a chart for numeric results or when your question asks for a chart.",
)
# Inform the user when charts are disabled.
if not charts_available():
    st.caption("Charts disabled until pandas and plotly are installed.")

# Load schema once for the prompt and safety checks.
try:
    schema = extract_schema(db_url)
    schema_details = json.dumps(schema, ensure_ascii=True)
except Exception as exc:
    st.error(f"Could not load schema: {exc}")
    st.stop()

# Sidebar schema browser.
st.sidebar.header("Database schema")
st.sidebar.caption("Click a table to see its columns.")
for table_name in sorted(schema.keys()):
    with st.sidebar.expander(table_name):
        st.markdown("\n".join(f"- {col}" for col in schema[table_name]))

# Initialize chat history once.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Ask questions about your data in plain language. "
                "Examples: \"How many orders were placed last month?\", "
                "\"What is the range of order date?\". "
                "If you want a chart, include words like \"plot\" or use the "
                "\"Plot results\" toggle."
            ),
        }
    ]

# Re-render chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Only render extra details for assistant messages.
        if message["role"] == "assistant":
            # Show prior rows if present.
            if message.get("rows") is not None:
                render_results(message["rows"], show_chart=message.get("show_chart", False))
            # Show prior error if present.
            if message.get("error"):
                st.error(message["error"])
            # Optionally show SQL last.
            if show_work and message.get("sql"):
                st.code(message["sql"], wrap_lines=True, language="sql")

user_prompt = st.chat_input("Ask about your data")
# Only process when the user submits a question.
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Render the just-submitted question in the current run.
    with st.chat_message("user"):
        st.markdown(user_prompt)
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

        # Surface errors before attempting to answer.
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
        plot_requested = show_chart or wants_chart(user_prompt)

        # Summarize results, then show data and optional SQL.
        answer_rows = rows[: int(get_setting("ANSWER_MAX_ROWS", "50"))]
        with st.spinner("Generating answer..."):
            try:
                range_answer = infer_range_answer(answer_rows)
                # Prefer a direct range explanation when detected.
                if range_answer:
                    answer = range_answer
                else:
                    answer = generate_answer(user_prompt, answer_rows, model)
            except Exception as exc:
                answer = f"I could not generate an answer: {exc}"
        st.markdown(answer)
        render_results(rows, show_chart=plot_requested)
        # SQL appears last when "Show your work" is enabled.
        if show_work:
            st.code(sql, wrap_lines=True, language="sql")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sql": sql,
                "rows": rows,
                "show_chart": plot_requested,
            }
        )
