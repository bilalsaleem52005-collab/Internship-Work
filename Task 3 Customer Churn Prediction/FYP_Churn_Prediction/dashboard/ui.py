# dashboard/ui.py
import streamlit as st

def hr():
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

def section_title(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<div class='muted'>{subtitle}</div>", unsafe_allow_html=True)