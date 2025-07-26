import streamlit as st
import os
import tempfile

from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent  
from mcp import create_mcp_message

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Agentic RAG Chatbot")


# --- File Upload ---
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload multiple files", accept_multiple_files=True)
docs = []
for file in uploaded_files:
    try:
        docs.append(file.read().decode("utf-8"))  # Read as UTF-8
    except UnicodeDecodeError:
        docs.append(file.read().decode("latin1"))  # fallback


# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "responder" not in st.session_state:
    st.session_state.responder = LLMResponseAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# On file upload
if uploaded_files:
    ingestor = IngestionAgent()
    trace_id = "streamlit-chat"
    msg_ingested = ingestor.handle({
        "trace_id": trace_id,
        "payload": {
            "docs": docs  
        }
    })

    retriever = RetrievalAgent()
    retriever.handle(msg_ingested)

    st.session_state.retriever = retriever
    st.success("✅ Documents parsed and indexed!")


# --- Chat Interface ---
st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question:")

if st.button("Ask") and query and st.session_state.retriever:
    trace_id = "streamlit-chat"
    query_msg = create_mcp_message("UI", "RetrievalAgent", "QUERY", trace_id, {"query": query})
    retrieval_response = st.session_state.retriever.handle(query_msg)

    final_answer = st.session_state.responder.handle(retrieval_response)

    # Show result
    st.markdown(f"### 🧠 Answer:\n{final_answer['answer']}")

    # Source Chunks
    with st.expander("📚 Source Context"):
        for i, chunk in enumerate(final_answer["source_chunks"]):
            st.markdown(f"**Chunk {i+1}:** {chunk.strip()}\n")

    # Save chat
    st.session_state.chat_history.append((query, final_answer["answer"]))

# Chat history
if st.session_state.chat_history:
    st.sidebar.markdown("### 🕒 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.sidebar.markdown(f"**Q:** {q}\n\n**A:** {a}")
