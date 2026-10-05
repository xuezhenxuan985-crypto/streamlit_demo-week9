"""
Streamlit Boilerplate codes
========================================

Run:
    uv add streamlit
    uv run streamlit run src/week9_streamlit_demo.py
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Home Credit Dashboard", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data"/  "processed_data_cube.csv"

CATEGORICAL_COLUMNS = [
    "MONTH_APPLIED",
    "YEAR_APPLIED",
    "NAME_CONTRACT_TYPE",
    "AGE_GROUP",
    "CREDIT_INCOME_RATIO_GROUP",
    "YEARS_EMPLOYED_GROUP",
    "AGE_GENDER_SEGMENT",
    "HOME_CAR_OWNERSHIP",
    "BURDEN_CAT"
]

METRIC_OPTIONS = {
    "Total Applications": "total_applications",
    "Default Rate": "default_rate",
    "Total Credit": "total_credit"
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def filter_data(df):
    st.sidebar.header("Filters")
    selections = {}

    for column in CATEGORICAL_COLUMNS:
        values = df[column].dropna().unique().tolist()
        values = sorted(values)
        option = st.sidebar.selectbox(
            f"{column}",
            ["All"] + values,
            index=0,
            key=f"filter_{column}"
        )
        selections[column] = option

    filtered = df.copy()
    for column, selected in selections.items():
        if selected != "All":
            filtered = filtered[filtered[column] == selected]

    return filtered, selections


def compute_metrics(df):
    total_applications = int(df["total_applications"].sum())
    total_defaults = int(df["total_defaults"].sum())
    total_credit = float(df["total_credit"].sum())
    default_rate = total_defaults / total_applications if total_applications else 0

    return {
        "total_applications": total_applications,
        "total_defaults": total_defaults,
        "total_credit": total_credit,
        "default_rate": default_rate
    }


def build_drilldown(df):
    st.sidebar.header("Drilldown")
    group_by = st.sidebar.selectbox("Group by", CATEGORICAL_COLUMNS, index=2)
    metric_name = st.sidebar.selectbox("Metric", list(METRIC_OPTIONS.keys()), index=0)
    metric_col = METRIC_OPTIONS[metric_name]

    group_df = (
        df.groupby(group_by)
        .agg(
            total_applications=("total_applications", "sum"),
            total_defaults=("total_defaults", "sum"),
            total_credit=("total_credit", "sum")
        )
        .assign(default_rate=lambda x: x["total_defaults"] / x["total_applications"].replace(0, 1))
        .reset_index()
    )

    if metric_col == "default_rate":
        group_df = group_df.sort_values(by=metric_col, ascending=False)
        fig = px.bar(
            group_df,
            x=group_by,
            y=metric_col,
            title=f"{metric_name} by {group_by}",
            labels={group_by: group_by, metric_col: metric_name},
            text=group_df[metric_col].map("{:.1%}".format)
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
    else:
        group_df = group_df.sort_values(by=metric_col, ascending=False)
        fig = px.bar(
            group_df,
            x=group_by,
            y=metric_col,
            title=f"{metric_name} by {group_by}",
            labels={group_by: group_by, metric_col: metric_name},
            text=metric_col
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
    return group_df, group_by, metric_name


def main():
    st.title("🏠 Home Credit Dashboard")
    st.markdown(
        "Use the left sidebar to filter every categorical field and drill into key metrics."
    )

    df = load_data()
    filtered_df, selections = filter_data(df)
    metrics = compute_metrics(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applications", f"{metrics['total_applications']:,}")
    col2.metric("Default Rate", f"{metrics['default_rate']:.1%}")
    col3.metric("Total Credit", f"${metrics['total_credit']:,.0f}")

    st.divider()

    st.subheader("Filtered trend overview")
    st.markdown(
        "The dashboard shows aggregated values after applying all selected categorical filters. "
        "Change the drilldown dimension to see how the metric distributes across categories."
    )

    drill_df, selected_group_by, selected_metric_name = build_drilldown(filtered_df)

    active_filters = [
        f"{column}: {value}"
        for column, value in selections.items()
        if value != "All"
    ]
    active_filters_text = ", ".join(active_filters) if active_filters else "All filters"
    current_view_text = (
        f"Current view — Filters: {active_filters_text}; "
        f"Drilldown: {selected_group_by}; Metric: {selected_metric_name}."
    )
    st.caption(current_view_text)

    st.subheader("Top groups")
    st.dataframe(drill_df, use_container_width=True)

    st.write("---")
    st.caption("Data loaded from processed_data_cube.csv")


if __name__ == "__main__":
    main()



