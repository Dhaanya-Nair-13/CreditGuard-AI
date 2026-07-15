
"""
profile.py
Customer profile panel for CreditGuard-AI.
"""

import streamlit as st


def _value(data, key, default="Not Available"):
    value = data.get(key, default)
    if value in ("", None):
        return default
    return value


def render_profile(customer: dict):
    """
    customer: dictionary returned by the API.
    Missing fields are displayed gracefully.
    """

    st.markdown(
        """
<div class="cg-card">
<div class="cg-card-title">Customer Profile</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("##### Identification")

        st.markdown(f"""
| Field | Value |
|------|------|
| Customer ID | {_value(customer,"customer_id")} |
| Gender | {_value(customer,"gender")} |
| Education | {_value(customer,"education")} |
| Housing | {_value(customer,"housing")} |
| Children | {_value(customer,"children")} |
""")

        st.markdown("##### Employment")

        st.markdown(f"""
| Field | Value |
|------|------|
| Employment Years | {_value(customer,"employment_years")} |
| Occupation | {_value(customer,"occupation")} |
| Organization | {_value(customer,"organization")} |
""")

    with right:
        st.markdown("##### Financial Information")

        st.markdown(f"""
| Field | Value |
|------|------|
| Income | {_value(customer,"income")} |
| Credit Amount | {_value(customer,"credit_amount")} |
| Annuity | {_value(customer,"annuity")} |
| Goods Price | {_value(customer,"goods_price")} |
""")

        st.markdown("##### Assessment")

        st.markdown(f"""
| Field | Value |
|------|------|
| Risk | {_value(customer,"risk")} |
| Probability | {customer.get("probability",0)*100:.2f}% |
| Decision | {_value(customer,"recommendation")} |
""")

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )
