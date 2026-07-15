
"""
theme.py
Enterprise theme configuration for CreditGuard-AI
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Colors:
    BACKGROUND="#F4F6F8"
    SURFACE="#FFFFFF"
    BORDER="#D9E2EC"
    PRIMARY="#1F4E79"
    PRIMARY_DARK="#163A5C"
    SUCCESS="#2E7D32"
    WARNING="#C77700"
    DANGER="#C62828"
    TEXT="#1F2937"
    MUTED="#6B7280"
    SHADOW="0 10px 30px rgba(15,23,42,.08)"

@dataclass(frozen=True)
class Typography:
    FONT="Inter"
    TITLE=34
    SUBTITLE=18
    CARD_TITLE=13
    KPI=32
    BODY=15
    SMALL=12

GLOBAL_CSS=f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
font-family:'Inter',sans-serif;
}}

.stApp {{
background:{Colors.BACKGROUND};
}}

.block-container {{
max-width:1500px;
padding-top:1.8rem;
}}

#MainMenu, footer, header {{
visibility:hidden;
}}

.cg-header {{
display:flex;
justify-content:space-between;
align-items:center;
margin-bottom:28px;
}}

.cg-title {{
font-size:{Typography.TITLE}px;
font-weight:700;
color:{Colors.TEXT};
margin:0;
}}

.cg-subtitle {{
color:{Colors.MUTED};
font-size:{Typography.SUBTITLE}px;
margin-top:4px;
}}

.cg-card {{
background:{Colors.SURFACE};
border:1px solid {Colors.BORDER};
border-radius:14px;
padding:22px;
box-shadow:{Colors.SHADOW};
}}

.cg-card-title {{
color:{Colors.MUTED};
text-transform:uppercase;
letter-spacing:.08em;
font-size:{Typography.CARD_TITLE}px;
font-weight:600;
}}

.cg-kpi {{
color:{Colors.TEXT};
font-size:{Typography.KPI}px;
font-weight:700;
margin-top:8px;
}}

.cg-caption {{
color:{Colors.MUTED};
font-size:{Typography.SMALL}px;
}}

.stButton button {{
background:{Colors.PRIMARY};
color:white;
border:none;
border-radius:8px;
height:42px;
font-weight:600;
}}

.stButton button:hover {{
background:{Colors.PRIMARY_DARK};
}}
</style>
"""

def load_theme(st):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
