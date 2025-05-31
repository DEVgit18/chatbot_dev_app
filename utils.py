import streamlit as st

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def format_response(role, content):
    return {"role": role, "content": content}
