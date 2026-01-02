import time

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableMap
)


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")


model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

print(" Groq summoned!")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Extracting knowledge")

# Load the knowledge base file
loader = TextLoader('Bhartiya nyaya samhita.txt', encoding='utf-8')
documents = loader.load()

# print(f"✅ Loaded knowledge base: {len(documents[0].page_content):,} characters")
# print(f"📊 File size: ~{len(documents[0].page_content) / 1024 / 1024:.2f} MB")

# Split into smaller chunks
# print("\n✂️ Splitting into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # Each chunk = ~1000 characters
    chunk_overlap=200,        # 200 characters overlap between chunks
    separators=[
        "################################",  # Category separators
        "================",                # Case separators
        "\n\n",                           # Paragraph breaks
        "\n",                             # Line breaks
        " ",                              # Spaces
        ""                                # Characters
    ]
)

chunks = text_splitter.split_documents(documents)

VECTOR_DB_PATH = "faiss_bns_index"

if os.path.exists(VECTOR_DB_PATH):
    print("📦 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("🧠 Creating new FAISS index from knowledge base...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print("💾 FAISS index saved to disk")


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5  # number of relevant chunks to retrieve
    }
)

print("🔎 Retriever is ready")


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal information assistant for Indian law.

Your task is to:
1. Answer the user's legal question clearly and accurately.
2. Where possible, provide one relevant illustrative case or example
   drawn from the provided knowledge base.

Source priority:
- Prefer the provided context from the Bhartiya Nyaya Samhita and related cases.
- If the context is insufficient, you may rely on general legal understanding.

Guidelines:
- Clearly indicate whether the main answer is based on the Bhartiya Nyaya Samhita
  or on general legal understanding.
- The illustrative case/example must come ONLY from the provided context.
- Do not invent or assume case names, facts, or section numbers.
- If no suitable case/example is present in the context, explicitly say so.
- Do not provide legal advice or definitive instructions.
- Explain concepts in clear, plain language suitable for a non-lawyer.

Context (statutory provisions and cases):
{context}

Question:
{question}

Answer format:

Legal Answer:
<your answer here>

Relevant Case / Example from Knowledge Base:
<brief description of one matching case or example, if available>
"""
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_pipeline = (
    RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | StrOutputParser()
)

def generate_response(user_query: str) -> str:
    if not user_query or not user_query.strip():
        return "Please provide a valid legal query."

    response = rag_pipeline.invoke(user_query)
    return response


if __name__ == "__main__":
    test_query = "Does verbal threat or intimidation amount to a criminal offence under Indian law?"
    print(generate_response(test_query))

