
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# ENV LOAD
# ----------------------------
load_dotenv()

# ----------------------------
# STREAMLIT UI TITLE
# ----------------------------
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺"
)

st.title("🩺 Medical RAG Chatbot ")

# ----------------------------
# LOAD VECTOR DB
# ----------------------------
db_faiss_path = "vectorstore/db_faiss"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    db_faiss_path,
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

# ----------------------------
# LLM
# ----------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# PROMPT
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful medical assistant chatbot.
        Answer only from the provided context.
        If the answer is not available in the context,
        say: 'I could not find that information in the provided medical documents.'
        """
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

parser = StrOutputParser()

# ----------------------------
# LCEL CHAIN
# ----------------------------
chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | prompt
    | model
    | parser
)

# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for msg in st.session_state.messages:

    avatar = "👨‍⚕️" if msg["role"] == "assistant" else "🧑"

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):
        st.write(msg["content"])

# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.chat_input(
    "Ask your medical question..."
)

if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message(
        "user",
        avatar="🧑"
    ):
        st.write(user_input)

    # Generate AI response
    with st.chat_message(
        "assistant",
        avatar="👨‍⚕️"
    ):
        with st.spinner("Doctor is reviewing your question..."):
            response = chain.invoke(user_input)

        st.write(response)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

