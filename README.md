# 🧠 Agentic RAG Chatbot: Multi-Format Document Q&A

## 🧩 Overview

This project implements an **Agentic RAG Chatbot** capable of answering user queries over diverse uploaded documents (PDF, PPTX, CSV, DOCX, TXT). The architecture follows an **agent-based design** with inter-agent communication structured through a lightweight, custom **Model Context Protocol (MCP)** message schema — a structured JSON envelope defined in this project, not Anthropic's MCP standard.

The chatbot allows users to:

- Upload multiple document formats
- Ask questions interactively
- Get responses grounded in the source documents
- See source chunks that led to each answer

Two variants were explored:

- **Streamlit UI (deployed version)** → Uses lightweight **FLAN-T5-base** for practical hosting
- **Colab Notebook (powerful version)** → Uses **Mistral-7B-Instruct** for robust answer quality (FLAN-T5-large was evaluated too, but exceeded available memory, so **FLAN-T5-base** is used as the lightweight fallback)

---

## 🏗️ Agentic Architecture

This system uses **three core agents**:

### 1. `IngestionAgent`

- Parses uploaded files
- Converts them into chunks (preserving semantic structure)
- Supports `.pdf`, `.pptx`, `.csv`, `.docx`, `.txt`, `.md`

### 2. `RetrievalAgent`

- Embeds all chunks using **MiniLM (SentenceTransformer)**
- Stores in **FAISS vector DB**
- Uses semantic search to retrieve top-k relevant chunks per query

### 3. `LLMResponseAgent`

- Forms a final prompt by using the top-k chunks
- Generates an answer using an LLM
- Returns the answer + the used chunks

Agents communicate via **MCP format**, e.g.:

```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent",
  "type": "CONTEXT_RESPONSE",
  "trace_id": "abc-123",
  "payload": {
    "retrieved_context": ["chunk1", "chunk2"],
    "query": "Summarize the Q2 performance"
  }
}
```

---

## 🧠 LLM Models Used

| Interface     | Model                 | Notes                                               |
| ------------- | --------------------- | --------------------------------------------------- |
| Streamlit UI  | `flan-t5-base`        | Lightweight, runs well locally                      |
| Notebook Core | `mistral-7B-instruct` | Excellent performance, used for deep QA evaluation  |
| Notebook Core | `flan-t5-large`       | Evaluated for quality; exceeded local memory, so `flan-t5-base` was used instead |
| Notebook Core | `flan-t5-base`        | Lightweight fallback actually run alongside Mistral   |

### 🔍 Why FLAN in Streamlit?

Mistral gave *amazing results*, but couldn’t run on local system due to memory constraints. So we switched to `flan-t5-base` for deployment. We clearly highlight this choice in the README and the Streamlit UI.

---

## 🧪 Example Results

📁 Test File (Text): Contains Q1, Q2, Q3 insights and Customer Feedback

### ✅ Questions Tested:

- What were the Q1 highlights?
- How did the company perform in Q2?
- What initiatives were taken to reduce churn in Q2?
- What are the company’s plans for Q3?
- How did customers respond to the referral program?

### 🎯 Sample Mistral Answer:

> **Q: Summarize the Q2 performance**
>
> "The company reduced churn by 10%, improved delivery TAT by 25%, and introduced voice-based ordering."

---

## 🖼️ Streamlit UI Snapshots

Here are some snapshots from the deployed chatbot:

| Feature Area        | Screenshot Name     |
|---------------------|---------------------|
| File Upload Section | `file_upload`       |
| Q2 Performance QA   | `q2_improvement`    |
| Q3 Plans Question   | `Q3_plan`           |
| Customer Feedback   | `feedback`          |
| Referral Response   | `referral`          |

📸 These screenshots are stored in the `screenshots/` folder inside the repo.

---

## 🧱 Tech Stack

- **LLMs**: Mistral-7B-Instruct, FLAN-T5-Base & Large
- **Embeddings**: MiniLM (via `sentence-transformers`)
- **Vector DB**: FAISS
- **Interface**: Streamlit + Colab
- **Document Parsing**: PyMuPDF, python-docx, python-pptx, csv, file I/O

---

## 📌 Challenges Faced

- Mistral model was too large to run on local for Streamlit but can be seen working properly in colab
- Parsing some `.pptx` files had shape/text edge cases
- FLAN-T5 sometimes gave noisy outputs on long context → mitigated with chunk control

---

## 🚀 Future Improvements

- Add a `CoordinatorAgent` to manage flow between agents
- Deploy Mistral via API (Hugging Face Inference or local GPU)
- Add query type classification (e.g., date-based, entity-based)
- Improve chunking logic with sentence-boundary awareness

---

## ✅ Summary

This Agentic RAG Chatbot showcases how agent-based architectures and structured context passing (MCP) can be combined with semantic retrieval and powerful LLMs to create document-grounded QA experiences — both via notebook and deployable web UI.

---

## 🎥 Demo Walkthrough

Watch the 5-minute video demo here:  
👉 [Loom Video](https://www.loom.com/share/bc8ae9c8d28f41b69a8c43a6c851ea7c?sid=61af7c33-2462-47f0-a4e3-68a678a552c4)

---

## 📥 Install Dependencies
Make sure to install required Python packages using:

```bash
pip install -r requirements.txt
```

---

## 📂 Repo Contents

```
📁 Agentic-RAG-Chatbot
├── app.py                # Streamlit interface
├── agents.py             # Ingestion, Retrieval, LLMResponse agents
├── mcp.py                # Message creation utility
├── test_files/           # All 3 test text files
├── Agentic_RAG_QA.ipynb  # Main evaluation notebook (Mistral + FLAN)
├── screenshots/          # App and result visuals
├── architecture.pptx     # Slides for architecture and system flow
└── README.md             # This file
```

---

**🧑‍💻 Made with effort and experimentation by [Divyansh Gautam](https://github.com/Divyansh-git10)**


