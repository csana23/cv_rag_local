import os

import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_LOCAL_URL") if os.getenv("LOCAL") == "yes" else os.getenv("CHATBOT_DOCKER_URL")
print("CHATBOT_URL", CHATBOT_URL)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about CVs loaded into a directory.
        The agent uses  retrieval-augment generation (RAG) with a chromadb backend.
        """
    )


st.title("CV RAG Chatbot")
st.info(
    """Ask me questions about your CV!"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "answer" in message.keys():
            st.markdown(message["answer"])

        if "context" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["context"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "answer": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, data=data)

        if response.status_code == 200:
            output_text = response.json()["answer"]
            context = response.json()["context"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            context = output_text

    st.chat_message("assistant").markdown(output_text)
    # st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": output_text,
            "context": context,
        }
    )
