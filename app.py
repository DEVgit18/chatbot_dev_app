# app.py
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage

from hf_chat import hf_chat
from groq_chat import groq_chat
from utils import init_session  # Optional

load_dotenv()
st.set_page_config(page_title="Multi-Model Chatbot", page_icon="🤖")
st.title("🧠 Multi-Model Chatbot")

# Initialize session state
init_session()

# Model selector
model = st.selectbox("Choose a model", [
    "Groq (LLaMA3-70B)",
    "Hugging Face (Flan-T5)"
])

# Chat input
user_input = st.chat_input("Say something...")

# On user message
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        try:
            if model == "Groq (LLaMA3-70B)":
                reply = groq_chat(st.session_state.messages)
            else:
                reply = hf_chat(st.session_state.messages)

            st.session_state.messages.append(AIMessage(content=reply))
        except Exception as e:
            st.session_state.messages.append(AIMessage(content=f"❌ Error: {e}"))

# Render chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Optional reset button
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
