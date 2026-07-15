
"""
cards.py
Enterprise dashboard card components.
"""

import streamlit as st


def begin_card():
    st.markdown('<div class="cg-card">', unsafe_allow_html=True)


def end_card():
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_card(title: str, value: str, caption: str = ""):
    st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">{title}</div>
  <div class="cg-kpi">{value}</div>
  <div class="cg-caption">{caption}</div>
</div>
""", unsafe_allow_html=True)


def status_badge(status: str):
    color = "#2E7D32"
    if status.lower().startswith("high"):
        color = "#C62828"
    elif status.lower().startswith("medium"):
        color = "#C77700"

    st.markdown(
        f"""
<div style="
display:inline-block;
padding:6px 12px;
border-radius:999px;
border:1px solid {color};
color:{color};
font-weight:600;
font-size:13px;
background:white;">
{status}
</div>
""",
        unsafe_allow_html=True,
    )


def executive_summary(probability: float, risk: str):
    if risk == "High Risk":
        text = (
            "The applicant demonstrates an elevated probability of default. "
            "The model indicates multiple adverse credit indicators. "
            "The application should be escalated according to current lending policy."
        )
    elif risk == "Medium Risk":
        text = (
            "The applicant falls within the review band. "
            "Additional underwriting checks are recommended before a lending decision."
        )
    else:
        text = (
            "The applicant demonstrates a comparatively low default probability. "
            "The application is suitable for the standard approval workflow."
        )

    st.markdown(f"""
<div class="cg-card">
<div class="cg-card-title">Executive Summary</div>
<p style="margin-top:14px;line-height:1.7;color:#374151;">
{text}
</p>
<hr style="border:none;border-top:1px solid #E5E7EB;">
<div class="cg-caption">
Default Probability: <b>{probability*100:.2f}%</b>
</div>
</div>
""", unsafe_allow_html=True)


def recommendation(probability: float):
    if probability >= 0.65:
        decision = "Reject Application"
    elif probability >= 0.35:
        decision = "Manual Review"
    else:
        decision = "Approve Application"

    st.markdown(f"""
<div class="cg-card">
<div class="cg-card-title">Recommendation</div>
<div class="cg-kpi" style="font-size:24px;">
{decision}
</div>
<div class="cg-caption">
Generated using the configured decision threshold.
</div>
</div>
""", unsafe_allow_html=True)


def model_information():
    st.markdown("""
<div class="cg-card">
<div class="cg-card-title">Model Information</div>

<table style="width:100%;margin-top:12px;">
<tr><td>Model</td><td><b>XGBoost (Optuna)</b></td></tr>
<tr><td>Explainability</td><td><b>SHAP</b></td></tr>
<tr><td>ROC-AUC</td><td><b>0.79</b></td></tr>
<tr><td>Customers</td><td><b>307,511</b></td></tr>
<tr><td>Features</td><td><b>530+</b></td></tr>
<tr><td>API</td><td><b>FastAPI</b></td></tr>
</table>

</div>
""", unsafe_allow_html=True)
