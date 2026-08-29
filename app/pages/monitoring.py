import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

password = st.text_input("Enter admin password", type="password")
if password != "19202007004409":
    st.error("Access denied")
    st.stop()

st.set_page_config(page_title="Monitoring", layout="wide")
st.title("Monitoring Dashboard")
st.caption("Real-time prediction analytics")

response = requests.get("https://web-production-e8e0e.up.railway.app/status")
data = response.json()["predictions"]


if not data:
    st.info("No predictions yet. Go analyze some audio!")
else:
    df = pd.DataFrame(data, columns=[
        "id", "timestamp", "prediction",
        "confidence", "latency_ms", "input_method"
    ])

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(df))
    col2.metric("Fake", len(df[df["prediction"] == "fake"]))
    col3.metric("Real", len(df[df["prediction"] == "real"]))
    col4.metric("Avg Latency", f"{df['latency_ms'].mean():.0f}ms")

    st.divider()

# Fake vs Real pie chart
    fig1 = px.pie(
        values=df["prediction"].value_counts().values,
        names=df["prediction"].value_counts().index,
        title="Fake vs Real Distribution",
        color_discrete_map={"fake": "#ff4b4b", "real": "#00cc44"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Input methods bar chart
    fig2 = px.bar(
        x=df["input_method"].value_counts().index,
        y=df["input_method"].value_counts().values,
        title="Predictions by Input Method",
        color_discrete_sequence=["#667eea"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Latency line chart
    fig3 = px.line(
        df, y="latency_ms",
        title="Latency Over Time (ms)",
        color_discrete_sequence=["#f093fb"]
    )
    st.plotly_chart(fig3, use_container_width=True)