
"""
summary.py
Executive summary and decision timeline components.
"""

import streamlit as st


def render_summary(result: dict):
    probability = result["probability"]
    risk = result["risk"]

    if risk == "High Risk":
        narrative = (
            "The applicant presents an elevated probability of default. "
            "The strongest contributors are concentrated around external credit "
            "history and repayment behaviour. Based on the configured policy "
            "threshold, the application should be escalated or declined."
        )
    elif risk == "Medium Risk":
        narrative = (
            "The applicant falls within the review band. Automated checks are "
            "inconclusive and additional underwriting is recommended before "
            "a lending decision is made."
        )
    else:
        narrative = (
            "The applicant demonstrates a comparatively low probability of "
            "default. Current indicators support the standard approval process "
            "subject to routine verification."
        )

    st.markdown(
        f"""
<div class="cg-card">
<div class="cg-card-title">Executive Summary</div>

<p style="margin-top:18px;line-height:1.8;color:#374151;">
{narrative}
</p>

<hr style="border:none;border-top:1px solid #E5E7EB;margin:20px 0;">

<table style="width:100%;font-size:14px;">
<tr>
<td style="color:#6B7280;">Risk Classification</td>
<td style="text-align:right;"><b>{risk}</b></td>
</tr>
<tr>
<td style="color:#6B7280;">Default Probability</td>
<td style="text-align:right;"><b>{probability*100:.2f}%</b></td>
</tr>
</table>

</div>
""",
        unsafe_allow_html=True,
    )


def render_timeline():
    st.markdown(
        """
<div class="cg-card">
<div class="cg-card-title">Assessment Workflow</div>

<div style="margin-top:20px;line-height:2.1;font-size:14px;">

<b>1.</b> Customer Record Retrieved<br>

↓<br>

<b>2.</b> Feature Store Lookup<br>

↓<br>

<b>3.</b> Model Inference (XGBoost)<br>

↓<br>

<b>4.</b> SHAP Attribution<br>

↓<br>

<b>5.</b> Credit Recommendation

</div>

</div>
""",
        unsafe_allow_html=True,
    )


def render_top_drivers(top_factors):
    st.markdown("#### Feature Attribution")

    for factor in top_factors:
        color = "#C62828" if factor["direction"] == "Increases Risk" else "#2E7D32"

        st.markdown(
            f"""
<div style="
display:flex;
justify-content:space-between;
padding:10px 0;
border-bottom:1px solid #E5E7EB;">

<span>{factor["feature"]}</span>

<span style="color:{color};font-weight:600;">
{factor["impact"]:+.4f}
</span>

</div>
""",
            unsafe_allow_html=True,
        )
