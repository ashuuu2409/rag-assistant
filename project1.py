# project1.py - Ready Tensor RAG Assistant with Google Gemini + Streamlit UI

import os
import json
import torch
import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --------------------------
# Setup
# --------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ No Google API key found. Please add GOOGLE_API_KEY to your .env file.")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./rt_publications_db")
collection = client.get_or_create_collection(
    name="rt_publications",
    metadata={"hnsw:space": "cosine"}
)

# Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# LLM (Google Gemini Free)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",  # ✅ free and fast
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# --------------------------
# Helper Functions
# --------------------------
def load_publications(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    publications = []
    for pub in data:
        title = pub.get("title", "Untitled")
        content = pub.get("publication_description", "")
        if content:
            publications.append((title, content))
    return publications

def chunk_publication(title, content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(content)
    return [{"content": c, "title": title, "chunk_id": f"{title}_{i}"} for i, c in enumerate(chunks)]

def insert_publications(collection, publications):
    if collection.count() > 0:
        return
    next_id = 0
    for title, content in publications:
        chunked = chunk_publication(title, content)
        texts = [c["content"] for c in chunked]
        metas = [{"title": c["title"], "chunk_id": c["chunk_id"]} for c in chunked]
        vectors = embeddings.embed_documents(texts)
        ids = [f"doc_{i}" for i in range(next_id, next_id + len(chunked))]
        collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        next_id += len(chunked)

def search_publications(query, collection, embeddings, top_k=3):
    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]
        }
        for i, doc in enumerate(results["documents"][0])
    ]

def answer_question(query):
    relevant = search_publications(query, collection, embeddings, top_k=3)
    context = "\n\n".join([f"From {c['title']}:\n{c['content']}" for c in relevant])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following publications, answer the question:

Context:
{context}

Question: {question}

Answer (grounded in the publications):
"""
    )
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant

# --------------------------
# Load Dataset into DB (only once)
# --------------------------
publications = load_publications("data/project_1_publications.json")
insert_publications(collection, publications)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="centered")

st.title("📚 Ready Tensor RAG Assistant")
st.write("Ask questions about the Ready Tensor publications dataset.")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... 🤔"):
            answer, sources = answer_question(query)
        st.subheader("🤖 AI Answer")
        st.write(answer)
        st.subheader("📌 Sources")
        for s in sources:
            st.markdown(f"- **{s['title']}** (similarity: {s['similarity']:.2f})")
