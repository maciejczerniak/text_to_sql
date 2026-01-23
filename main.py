import base64
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

# Text-to-SQL pipeline:
# 1) Build schema context from the database.
# 2) Ask the model for SQL (and fix it on errors).
# 3) Execute the query, summarize the results, and render tables/charts.

import streamlit as st
import streamlit.components.v1 as components
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
    return bool(
        re.search(
            r"\b(plot|chart|graph|visuali[sz]e|trend|line|bar|histogram|scatter|pie|draw|diagram)\b",
            question,
            re.I,
        )
    )


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


def needs_clarification(question: str, schema: Dict[str, List[str]]) -> Optional[str]:
    """Return a clarification prompt when the question is too vague."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
    if len(tokens) < 3:
        return "Could you clarify your question with more detail?"
    schema_terms = {name.lower() for name in schema.keys()}
    for cols in schema.values():
        schema_terms.update(col.lower() for col in cols)
    mentions_schema = any(tok in schema_terms for tok in tokens)
    generic = {"show", "list", "data", "info", "details", "report", "stats", "summary", "everything", "all", "overview"}
    has_generic = any(tok in generic for tok in tokens)
    has_filter = any(
        tok in {"count", "average", "avg", "min", "max", "sum", "total", "range", "between", "before", "after", "since",
                "during", "latest", "last", "top", "bottom", "most", "least"}
        for tok in tokens
    ) or bool(re.search(r"\d", question))
    if has_generic and not mentions_schema and not has_filter:
        table_names = sorted(schema.keys())[:3]
        if table_names:
            examples = ", ".join(table_names)
            return (
                "Could you clarify what you want to see? "
                f"For example, ask about {examples}, or specify a metric and time range."
            )
        return "Could you clarify what you want to see? Please specify a metric and any filters."
    return None


def render_sql_button(sql: str, index: int) -> None:
    """Show SQL without triggering a rerun that cancels in-flight work."""
    with st.expander("Show SQL"):
        st.code(sql, wrap_lines=True, language="sql")

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


DEFAULT_GREETING = {
    "role": "assistant",
    "content": (
        "Hi! Ask questions about your data in plain language. "
        "Examples: \"How many orders were placed last month?\", "
        "\"What is the range of order date?\". "
        "If you want a chart, include words like \"plot\" or use the "
        "\"Plot results\" toggle. Use the \"Show SQL\" button to see "
        "the query behind each answer."
    ),
}


st.set_page_config(page_title="Query Assistant", initial_sidebar_state="expanded", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --header-height: 80px;
        --footer-height: 80px;
        --input-height: 90px;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    div[data-testid="stDecoration"],
    div[data-testid="stToolbar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    button[title="Close sidebar"],
    button[title="Open sidebar"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none;
    }
    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        background: var(--background-color, white);
        border-bottom: 1px solid #ddd;
        padding: 10px 20px;
    }
    .app-title {
        font-size: 20px;
        font-weight: 600;
        margin: 0;
    }
    .app-meta {
        color: rgba(0, 0, 0, 0.6);
        font-size: 0.85rem;
    }
    .content {
        margin-top: var(--header-height);
    }
    .chat-scroll {
        overflow-y: auto;
        max-height: calc(100vh - var(--header-height) - var(--footer-height) - var(--input-height));
        padding-right: 8px;
    }
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        z-index: 1000;
        padding: 10px 20px;
        border-top: 1px solid #ddd;
        text-align: center;
    }
    section.main > div {
        margin-bottom: 80px;
        box-sizing: border-box;
    }
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: var(--footer-height);
        background: var(--background-color, white);
        padding-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

db_display_name = get_db_display_name(db_url)
header_container = st.container()
with header_container:
    st.markdown('<div id="header-anchor"></div>', unsafe_allow_html=True)
    header_left, header_center, header_right = st.columns([1.2, 3, 1.2], vertical_alignment="center")
    with header_left:
        st.markdown('')
    with header_center:
        st.markdown('<div class="app-title">Query Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="app-meta">Model: {OLLAMA_MODEL} | Database: {db_display_name}</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        show_chart = st.toggle(
            "Plot results",
            value=False,
            key="show_chart",
            help="Render a chart for numeric results or when your question asks for a chart.",
        )
        if not charts_available():
            st.caption("Charts disabled until pandas and plotly are installed.")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [DEFAULT_GREETING]
            st.rerun()

st.markdown(
    """
    <div class="fixed-footer">
        <p>© 2026 My App</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load schema once for the prompt and safety checks.
try:
    schema = extract_schema(db_url)
    schema_details = json.dumps(schema, ensure_ascii=True)
except Exception as exc:
    st.error(f"Could not load schema: {exc}")
    st.stop()

# Sidebar logo at the top.
logo_path = os.path.join(os.path.dirname(__file__), "img", "alternate.png")
try:
    with open(logo_path, "rb") as handle:
        logo_b64 = base64.b64encode(handle.read()).decode("utf-8")
    st.sidebar.markdown(
        f"""
        <style>
        .sidebar-logo {{
            text-align: center;
            margin: 4px 0 12px;
        }}
        .sidebar-logo img {{
            width: 100%;
            height: auto;
            border: 0px solid #4FC3F7;
        }}
        </style>
        <div class="sidebar-logo">
            <a href="https://alternate.nl" target="_blank" rel="noopener noreferrer">
                <img src="data:image/png;base64,{logo_b64}" alt="alternate.nl" />
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    pass

# Sidebar schema browser.
st.sidebar.header("Database schema")
st.sidebar.caption("Click a table to see its columns.")
for table_name in sorted(schema.keys()):
    with st.sidebar.expander(table_name):
        st.markdown("\n".join(f"- {col}" for col in schema[table_name]))

chat_container = st.container()
with chat_container:
    st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)
    # Initialize chat history once.
    if "messages" not in st.session_state:
        st.session_state.messages = [DEFAULT_GREETING]

    # Re-render chat history.
    for idx, message in enumerate(st.session_state.messages):
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
                # Optional SQL reveal.
                if message.get("sql"):
                    render_sql_button(message["sql"], idx)

components.html(
    """
    <script>
    const start = Date.now();
    const timer = setInterval(() => {
        const headerAnchor = window.parent.document.getElementById("header-anchor");
        if (headerAnchor) {
            const block = headerAnchor.closest('div[data-testid="stVerticalBlock"]');
            if (block && !block.classList.contains("app-header")) {
                block.classList.add("app-header");
            }
        }
        const chatAnchor = window.parent.document.getElementById("chat-scroll-anchor");
        if (chatAnchor) {
            const block = chatAnchor.closest('div[data-testid="stVerticalBlock"]');
            if (block && !block.classList.contains("chat-scroll")) {
                block.classList.add("chat-scroll");
            }
            if (block && !block.classList.contains("content")) {
                block.classList.add("content");
            }
        }
        if ((headerAnchor && chatAnchor) || Date.now() - start > 3000) {
            clearInterval(timer);
        }
    }, 50);
    </script>
    """,
    height=0,
)

user_prompt = st.chat_input("Ask about your data")
# Only process when the user submits a question.
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Render the just-submitted question in the current run.
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_prompt)
    clarification = needs_clarification(user_prompt, schema)
    if clarification:
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        st.stop()
    with chat_container:
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
            render_sql_button(sql, len(st.session_state.messages))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sql": sql,
                    "rows": rows,
                    "show_chart": plot_requested,
                }
            )
