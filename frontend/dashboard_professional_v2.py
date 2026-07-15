
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CreditGuard-AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    font-family:'Inter',sans-serif;
}
.main{
    background:#F8FAFC;
}
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}
h1{
    color:#0f172a;
}
.card{
    background:white;
    border-radius:10px;
    padding:22px;
    border:1px solid #E5E7EB;
    box-shadow:0 2px 6px rgba(15,23,42,.06);
}
.footer{
    text-align:center;
    color:gray;
    padding:30px;
}
</style>
""", unsafe_allow_html=True)

st.title("CreditGuard-AI")
st.caption("AI-powered Loan Default Risk Assessment Platform")

col1,col2=st.columns([4,1])

with col1:
    customer_id=st.number_input(
        "Customer ID",
        min_value=100000,
        value=100002,
        step=1
    )

with col2:
    st.write("")
    st.write("")
    analyze=st.button("Run Assessment", use_container_width=True)

if analyze:

    with st.spinner("Running model..."):
        r=requests.get(f"{API_URL}/customers/{customer_id}")

    if r.status_code!=200:
        st.error("Customer not found.")
        st.stop()

    result=r.json()

    prob=result["probability"]
    risk=result["risk"]

    if prob>=0.65:
        recommendation="Reject Loan"
    elif prob>=0.35:
        recommendation="Manual Review"
    else:
        recommendation="Approve"

    confidence=round(abs(prob-0.5)*200,1)

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Probability",f"{prob*100:.2f}%")
    c2.metric("Risk",risk)
    if recommendation == "Reject Loan":
        c3.markdown(f"""
<div class="card">
<div style="font-size:13px;color:#6B7280;font-weight:600;">Recommendation</div>
<div style="font-size:30px;font-weight:700;color:#DC2626;margin-top:8px;">
{recommendation}
</div>
</div>
""", unsafe_allow_html=True)
    elif recommendation == "Approve":
        c3.markdown(f"""
<div class="card">
<div style="font-size:13px;color:#6B7280;font-weight:600;">Recommendation</div>
<div style="font-size:30px;font-weight:700;color:#16A34A;margin-top:8px;">
{recommendation}
</div>
</div>
""", unsafe_allow_html=True)
    else:
        c3.markdown(f"""
<div class="card">
<div style="font-size:13px;color:#6B7280;font-weight:600;">Recommendation</div>
<div style="font-size:30px;font-weight:700;color:#D97706;margin-top:8px;">
{recommendation}
</div>
</div>
""", unsafe_allow_html=True)
    c4.metric("Confidence",f"{confidence:.1f}%")

    st.divider()

    left,right=st.columns([1,1])

    with left:
        gauge=go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={'suffix':'%'},
            title={'text':'Default Probability'},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'darkred'},
                'steps':[
                    {'range':[0,35],'color':'#10B981'},
                    {'range':[35,65],'color':'#F59E0B'},
                    {'range':[65,100],'color':'#EF4444'}
                ]
            }
        ))
        gauge.update_layout(height=420)
        st.plotly_chart(gauge,use_container_width=True)

    with right:
        st.subheader("Executive Summary")

        if risk=="High Risk":
            st.error(
                "The applicant exhibits a high predicted probability of loan default. "
                "Multiple credit-related indicators contribute positively to risk. "
                "Recommendation: Reject the application or escalate for senior credit review."
            )
        elif risk=="Medium Risk":
            st.warning(
                "The applicant falls into the medium-risk segment. "
                "A manual underwriting review is recommended before approval."
            )
        else:
            st.success(
                "The applicant demonstrates a relatively low probability of default. "
                "The application may proceed subject to standard verification."
            )

        st.markdown("### Top Drivers")

        for f in result["top_factors"][:5]:
            st.write(f"**{f['feature']}** ({f['impact']:+.4f})")

    st.divider()

    df=pd.DataFrame(result["top_factors"])
    df["abs"]=df["impact"].abs()
    df=df.sort_values("abs")

    colors=["#EF4444" if x>0 else "#10B981" for x in df["impact"]]

    fig=go.Figure(go.Bar(
        x=df["impact"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        text=df["impact"],
        textposition="outside"
    ))

    fig.update_layout(
        title="Feature Importance (SHAP)",
        xaxis_title="SHAP Impact",
        yaxis_title="",
        height=500
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("""
<div class="footer">
Built with FastAPI • XGBoost • SHAP • Streamlit • Optuna
</div>
""",unsafe_allow_html=True)
