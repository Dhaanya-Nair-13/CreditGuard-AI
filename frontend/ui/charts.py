
"""
charts.py
Professional Plotly visualizations for CreditGuard-AI.
"""

import pandas as pd
import plotly.graph_objects as go

PRIMARY = "#1F4E79"
SUCCESS = "#2E7D32"
WARNING = "#C77700"
DANGER = "#C62828"
GRID = "#E5E7EB"


def probability_gauge(probability: float):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": "Default Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PRIMARY},
            "steps": [
                {"range": [0, 35], "color": SUCCESS},
                {"range": [35, 65], "color": WARNING},
                {"range": [65, 100], "color": DANGER},
            ],
        },
    ))

    fig.update_layout(
        height=360,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def confidence_gauge(probability: float):

    confidence = abs(probability - 0.5) * 200

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%"},
        title={"text": "Model Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PRIMARY},
            "steps": [
                {"range": [0, 50], "color": "#E5E7EB"},
                {"range": [50, 75], "color": WARNING},
                {"range": [75, 100], "color": SUCCESS},
            ],
        },
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def shap_chart(top_factors):

    df = pd.DataFrame(top_factors)

    df["absolute"] = df["impact"].abs()

    df = df.sort_values("absolute")

    colors = [
        DANGER if x > 0 else SUCCESS
        for x in df["impact"]
    ]

    fig = go.Figure(go.Bar(
        x=df["impact"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        text=df["impact"].round(3),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Impact=%{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Feature Attribution",
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title="SHAP Contribution",
            gridcolor=GRID,
            zeroline=True,
            zerolinecolor=GRID,
        ),
        yaxis=dict(title=""),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def risk_distribution(probability: float):

    values = [
        probability * 100,
        100 - (probability * 100),
    ]

    fig = go.Figure(go.Pie(
        labels=[
            "Default Risk",
            "Approval Margin",
        ],
        values=values,
        hole=0.72,
        marker=dict(
            colors=[
                DANGER,
                SUCCESS,
            ]
        ),
        textinfo="label+percent",
    ))

    fig.update_layout(
        title="Decision Distribution",
        height=320,
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig
