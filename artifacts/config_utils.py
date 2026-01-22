import os
import re
import urllib.parse
from typing import Optional

import streamlit as st


# Lightweight .env loader to avoid adding a dependency.
def load_dotenv_file(path: str = ".env", override: bool = False) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            # Read .env line-by-line.
            for raw_line in handle:
                line = raw_line.strip()
                # Skip blanks and comments.
                if not line or line.startswith("#"):
                    continue
                key, sep, value = line.partition("=")
                # Ignore malformed lines without a key=value separator.
                if not sep:
                    continue
                key = key.strip()
                value = value.strip()
                # Remove surrounding quotes if present.
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                # Keep existing env values unless override=True.
                if not override and key in os.environ:
                    continue
                os.environ[key] = value
    except FileNotFoundError:
        pass


# Prefer Streamlit secrets; fall back to environment variables.
def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve config from st.secrets first, then environment variables."""
    try:
        # Use Streamlit secrets when available.
        if key in st.secrets:
            return st.secrets.get(key)
    except Exception:
        pass
    return os.getenv(key, default)


# Build a SQLAlchemy URL from DB_URL or individual settings.
def build_db_url() -> Optional[str]:
    """Return a SQLAlchemy connection URL or None if settings are incomplete."""
    db_url = get_setting("DB_URL")
    # Prefer a fully specified DB_URL when provided.
    if db_url:
        return db_url

    dialect = get_setting("DB_DIALECT", "sqlite")
    # SQLite uses a file path instead of host/user/password.
    if dialect.startswith("sqlite"):
        return get_setting("SQLITE_PATH", "sqlite:///testdb.sqlite")

    host = get_setting("DB_HOST")
    user = get_setting("DB_USER")
    password = get_setting("DB_PASSWORD", "")
    name = get_setting("DB_NAME")
    port = get_setting("DB_PORT", "")

    # Bail out if required settings are missing.
    if not host or not user or not name:
        return None

    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password) if password else ""
    auth = f"{user_enc}:{pass_enc}@" if password else f"{user_enc}@"
    hostport = f"{host}:{port}" if port else host
    return f"{dialect}://{auth}{hostport}/{name}"


# Redact passwords before printing.
def mask_db_url(url: str) -> str:
    """Hide DB password when showing the URL in the UI."""
    return re.sub(r":([^:@/]+)@", ":****@", url)
