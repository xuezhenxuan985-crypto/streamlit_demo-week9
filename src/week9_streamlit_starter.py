"""
Streamlit Barebone Starter -- Week 9
========================================
Clone this repo, then run:
    uv run streamlit run src/week9_streamlit_starter.py

This is intentionally minimal. In class we will build it up together:
    - add sidebar filters
    - add KPI metrics
    - add a drilldown chart with a dimension/metric picker
    - publish it to Streamlit Community Cloud

Data source: processed_data_cube.csv, produced by running main.py
(the cube is created by create_cubes() in stage_3_aggregate.py).
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Home Credit Dashboard", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data"/  "processed_data_cube.csv"

DIMENSIONS = {
    "Contract type": "NAME_CONTRACT_TYPE",
    "Age group": "AGE_GROUP",
    "Credit-to-income ratio": "CREDIT_INCOME_RATIO_GROUP",
    "Years employed": "YEARS_EMPLOYED_GROUP",
    "Age and gender segment": "AGE_GENDER_SEGMENT",
    "Home / car ownership": "HOME_CAR_OWNERSHIP",
    "Debt burden": "BURDEN_CAT",
}

METRICS = {
    "Total applications": "total_applications",
    "Total defaults": "total_defaults",
    "Total credit": "total_credit",
    "Recent previous applications": "total_recent_prev_apps",
    "Refused previous applications": "total_refused_prev_apps",
    "Bureau loans": "total_bureau_loans",
    "Bureau debt": "total_bureau_debt",
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["month_applied"] = pd.to_datetime({
        "year": df["YEAR_APPLIED"],
        "month": df["MONTH_APPLIED"],
        "day": 1,
    })
    return df


def main():
    st.title("Home Credit Dashboard")
    st.markdown("Starter dashboard -- we'll build this up together in class.")

    df = load_data()

    month_min = df["month_applied"].min().to_pydatetime()
    month_max = df["month_applied"].max().to_pydatetime()

    with st.sidebar:
        from_month, to_month = st.slider(
            "Application month range",
            min_value=month_min,
            max_value=month_max,
            value=(month_min, month_max),
            format="MMM YYYY",
        )

    filtered = df[(df["month_applied"] >= from_month) & (df["month_applied"] <= to_month)]

    kpi_sums = filtered[["total_applications", "total_defaults", "total_credit"]].sum()
    total_apps_k = kpi_sums["total_applications"] / 1_000
    default_rate_pct = kpi_sums["total_defaults"] / kpi_sums["total_applications"] * 100
    total_credit_m = kpi_sums["total_credit"] / 1e6

    st.subheader("Key metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total applications", f"{total_apps_k:,.1f}K", border=True)
    with col2:
        st.metric("Default rate", f"{default_rate_pct:.2f}%", border=True)
    with col3:
        st.metric("Total credit", f"{total_credit_m:,.0f}M", border=True)

    st.subheader("Data cube preview")
    st.dataframe(filtered.head(20), width = 'stretch')

    st.subheader("Applications by contract type")

    chart_df = (filtered
                .groupby("NAME_CONTRACT_TYPE")
                .agg({"total_applications": "sum"})
                .reset_index()
                )

    fig = px.bar(chart_df, 
                 x="NAME_CONTRACT_TYPE",
                 y="total_applications")

    st.plotly_chart(fig, width= 'stretch')

    st.subheader("Drilldown: pick a dimension and a metric")

    picker1, picker2 = st.columns(2)
    with picker1:
        dimension_label = st.selectbox("Dimension", list(DIMENSIONS))
    with picker2:
        metric_label = st.selectbox("Metric", list(METRICS))

    dimension = DIMENSIONS[dimension_label]
    metric = METRICS[metric_label]

    drilldown_df = (filtered
                    .groupby(dimension)
                    .agg({metric: "sum"})
                    .reset_index()
                    .sort_values(metric, ascending=False)
                    )

    drilldown_fig = px.bar(drilldown_df,
                           x=dimension,
                           y=metric,
                           title=f"{metric_label} by {dimension_label}",
                           labels={dimension: dimension_label, metric: metric_label})

    st.plotly_chart(drilldown_fig, width="stretch")


if __name__ == "__main__":
    main()
