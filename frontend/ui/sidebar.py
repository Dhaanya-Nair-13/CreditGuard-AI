
"""
sidebar.py
Enterprise sidebar for CreditGuard-AI
"""

import streamlit as st
from datetime import datetime


def render_sidebar():

    with st.sidebar:

        st.markdown(
            """
<div style="padding:10px 0 18px 0;">
<h2 style="margin:0;color:#1F2937;font-weight:700;">
CreditGuard-AI
</h2>
<div style="color:#6B7280;font-size:13px;">
Loan Default Risk Assessment
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("### Navigation")

        pages = [
            "Dashboard",
            "Predictions",
            "Model Performance",
            "Documentation",
            "Settings",
        ]

        selected = st.radio(
            "",
            pages,
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown("### Environment")

        col1, col2 = st.columns([1, 5])

        with col1:
            st.markdown(
                "<div style='width:10px;height:10px;background:#2E7D32;border-radius:50%;margin-top:8px;'></div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                "<span style='font-size:14px;'>Production</span>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        st.caption(
            datetime.now().strftime(
                "%d %B %Y"
            )
        )

        st.markdown("---")

        st.markdown("### About")

        st.markdown(
            """
CreditGuard-AI is an explainable
credit risk assessment platform
powered by XGBoost and SHAP.

Version 1.0
"""
        )

        st.markdown("---")

        st.caption("© 2026 CreditGuard-AI")

    return selected
