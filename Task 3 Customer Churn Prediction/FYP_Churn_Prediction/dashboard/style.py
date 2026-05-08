# dashboard/style.py
import streamlit as st

LIGHT = """
<style>
:root{
  --bg: #F7FAFC;                 /* very light */
  --text: #0B1220;               /* very dark */
  --muted: rgba(11,18,32,0.68);
  --line: rgba(11,18,32,0.14);   /* divider line */
  --primary: #2563EB;
}

html, body, [class*="css"]{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  color: var(--text) !important;
}

/* Force readable text everywhere (fix invisible text in Light) */
.stMarkdown, .stMarkdown *,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
h1,h2,h3,h4,h5,h6, p, li, span, div, label, small{
  color: var(--text) !important;
}

.stApp{ background: var(--bg) !important; }

.block-container{
  max-width: 1220px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}

/* Sidebar (flat) */
section[data-testid="stSidebar"]{
  background: #FFFFFF !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

/* Remove ALL "boxy tiles" Streamlit adds */
div[data-testid="stMetric"]{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}
.stPlotlyChart, .element-container{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* Inputs (simple) */
div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea{
  background: #FFFFFF !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;        /* small radius */
  color: var(--text) !important;
}

/* Buttons (clean) */
.stButton button{
  background: var(--primary) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 8px !important;
  padding: 0.6rem 0.9rem !important;
  font-weight: 700 !important;
}
.stButton button:hover{ opacity: .95; }

/* Minimal divider utility */
.hr{
  height:1px;
  background: var(--line);
  margin: 14px 0 16px 0;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

DARK = """
<style>
:root{
  --bg: #0B1220;
  --text: #EAF0FF;
  --muted: rgba(234,240,255,0.72);
  --line: rgba(234,240,255,0.14);
  --primary: #3B82F6;
}

html, body, [class*="css"]{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  color: var(--text) !important;
}

.stMarkdown, .stMarkdown *,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
h1,h2,h3,h4,h5,h6, p, li, span, div, label, small{
  color: var(--text) !important;
}

.stApp{ background: var(--bg) !important; }

.block-container{
  max-width: 1220px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}

section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.03) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

div[data-testid="stMetric"]{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}
.stPlotlyChart, .element-container{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}

.stButton button{
  background: var(--primary) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 8px !important;
  padding: 0.6rem 0.9rem !important;
  font-weight: 700 !important;
}
.stButton button:hover{ opacity: .95; }

.hr{
  height:1px;
  background: var(--line);
  margin: 14px 0 16px 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

def apply_theme(theme: str):
    st.markdown(LIGHT if theme == "Light" else DARK, unsafe_allow_html=True)

def theme_selector():
    """
    Put this once at the top of EACH page.
    It will keep theme consistent across page changes.
    """
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    with st.sidebar:
        choice = st.radio(
            "🎨 Theme",
            ["Light", "Dark"],
            index=0 if st.session_state["theme"] == "Light" else 1,
            key="theme_choice"
        )
        st.session_state["theme"] = choice

    apply_theme(st.session_state["theme"])

# ---- BACKWARD COMPATIBILITY (so old pages don't break) ----
def apply_custom_style():
    theme = st.session_state.get("theme", "Light")
    apply_theme(theme)