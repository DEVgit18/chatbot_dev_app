# groq_chat.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def groq_chat(messages):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )
    response = llm.invoke(messages)
    return response.content
