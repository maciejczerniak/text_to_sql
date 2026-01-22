from typing import Any, Dict, List

import streamlit as st

# Optional deps for nicer tables/charts.
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
except ImportError:
    px = None


def charts_available() -> bool:
    """Return True when both pandas and plotly are installed."""
    return pd is not None and px is not None


# Detect datetime-like columns for better charts.
def coerce_datetime_columns(df):
    """Convert object columns to datetimes when they look date-like."""
    # Skip conversion if pandas is not available.
    if pd is None:
        return df
    # Inspect each column for datetime-like values.
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            # Treat column as datetime if most values parse cleanly.
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
    return df


# Pick a reasonable chart (line for time series, bar otherwise).
def build_chart(df):
    """Build a plotly chart from the dataframe when possible."""
    # Charts require pandas, plotly, and at least one row.
    if pd is None or px is None or df.empty:
        return None
    df = coerce_datetime_columns(df.copy())
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # No numeric columns means no sensible chart.
    if not numeric_cols:
        return None

    x_col = None
    # Pick the first non-numeric column as the x-axis.
    for col in df.columns:
        if col not in numeric_cols:
            x_col = col
            break
    # Fall back to a synthetic index when all columns are numeric.
    if x_col is None:
        df = df.reset_index(drop=True)
        df["index"] = df.index
        x_col = "index"

    # Line charts for time series; otherwise use a bar chart.
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        return px.line(df, x=x_col, y=numeric_cols)
    y_cols = numeric_cols if len(numeric_cols) > 1 else numeric_cols[0]
    return px.bar(df, x=x_col, y=y_cols)


# Render tabular data plus an optional chart.
def render_results(rows: List[Dict[str, Any]], show_chart: bool = False) -> None:
    """Display results as a table and an optional chart."""
    # Show rows if we have any.
    if rows:
        # Without pandas, render the list of dicts directly.
        if pd is None:
            st.dataframe(rows, use_container_width=True)
            return
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        # Only render a chart when explicitly requested.
        if show_chart:
            fig = build_chart(df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rows returned.")
