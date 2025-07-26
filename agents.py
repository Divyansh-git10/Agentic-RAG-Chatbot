# agents.py (FLAN-T5 base version)

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FLAN-T5 base model
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
flan_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


class IngestionAgent:
    def __init__(self, name="IngestionAgent"):
        self.name = name

    def handle(self, message):
        trace_id = message["trace_id"]
        docs = message["payload"]["docs"]

        chunks = []
        for doc in docs:
            lines = doc.split("\n")
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) < 500:
                    chunk += line + "\n"
                else:
                    chunks.append(chunk.strip())
                    chunk = line + "\n"
            if chunk:
                chunks.append(chunk.strip())

        return {
            "sender": self.name,
            "receiver": "RetrievalAgent",
            "type": "CONTEXT_CHUNKS",
            "trace_id": trace_id,
            "payload": {"chunks": chunks},
        }


class RetrievalAgent:
    def __init__(self, name="RetrievalAgent"):
        self.name = name
        self.model = embedding_model
        self.index = None
        self.chunk_map = []

    def build_index(self, chunks):
        embeddings = self.model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        self.chunk_map = chunks

    def handle(self, message):
        trace_id = message["trace_id"]

        if message["type"] == "CONTEXT_CHUNKS":
            chunks = message["payload"]["chunks"]
            self.build_index(chunks)
            return {
                "sender": self.name,
                "receiver": "LLMResponseAgent",
                "type": "READY",
                "trace_id": trace_id,
                "payload": {"status": "index_built"},
            }

        elif message["type"] == "QUERY":
            query = message["payload"]["query"]
            query_embedding = self.model.encode([query])
            D, I = self.index.search(np.array(query_embedding), k=3)
            top_chunks = [self.chunk_map[i] for i in I[0]]
            return {
                "sender": self.name,
                "receiver": "LLMResponseAgent",
                "type": "CONTEXT_RESPONSE",
                "trace_id": trace_id,
                "payload": {
                    "retrieved_context": top_chunks,
                    "query": query,
                },
            }


class LLMResponseAgent:
    def __init__(self, name="LLMResponseAgent"):
        self.name = name
        self.llm = flan_pipe

    def handle(self, message):
        context = "\n".join(message["payload"]["retrieved_context"])
        query = message["payload"]["query"]

        prompt = (
            f"You are a helpful business assistant. Use only the context below to answer the question clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )

        result = self.llm(prompt, max_new_tokens=300)
        return {
            "answer": result[0]["generated_text"].strip(),
            "source_chunks": message["payload"]["retrieved_context"],
        }
