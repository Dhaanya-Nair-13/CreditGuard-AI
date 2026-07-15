
"""
animations.py
Reusable UI helpers and subtle loading animations.
"""

import time
import streamlit as st


def inject_animation_css():
    st.markdown("""
<style>

.fade-in{
    animation:fadeIn .45s ease-in-out;
}

@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(10px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}

.cg-divider{
    height:1px;
    background:#E5E7EB;
    margin:22px 0;
}

.cg-pill{
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    border:1px solid #D1D5DB;
    font-size:12px;
    color:#4B5563;
    background:#FFFFFF;
}

</style>
""", unsafe_allow_html=True)


def page_header(title: str, subtitle: str):
    st.markdown(
        f"""
<div class="fade-in">
    <h1 style="
        margin-bottom:4px;
        font-size:34px;
        font-weight:700;
        color:#1F2937;">
        {title}
    </h1>

    <div style="
        color:#6B7280;
        font-size:15px;">
        {subtitle}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def loading(message="Running assessment..."):
    return st.spinner(message)


def progress(message="Loading model"):
    bar = st.progress(0, text=message)

    for i in range(101):
        time.sleep(0.003)
        bar.progress(i, text=message)

    bar.empty()


def section(title: str):
    st.markdown(
        f"""
<div class="fade-in">

<h3 style="
margin-top:10px;
margin-bottom:18px;
font-size:18px;
font-weight:600;
color:#1F2937;">
{title}
</h3>

</div>
""",
        unsafe_allow_html=True,
    )


def horizontal_rule():
    st.markdown(
        '<div class="cg-divider"></div>',
        unsafe_allow_html=True,
    )


def environment_badge():
    st.markdown(
        """
<span class="cg-pill">
Production Environment
</span>
""",
        unsafe_allow_html=True,
    )
