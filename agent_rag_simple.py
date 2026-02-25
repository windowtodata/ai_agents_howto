import argparse
import os

import bs4
import faiss
from langchain.agents import AgentState, create_agent
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH = os.path.expanduser("~/workspace/hf_cache/faiss_index")
BLOG_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# ---------------------------------------------------------------------------
# Embeddings (shared)
# ---------------------------------------------------------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ---------------------------------------------------------------------------
# Option 1 – FAISS persisted to disk
# ---------------------------------------------------------------------------
def init_faiss_vector_store(persist_path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Load an existing FAISS index from *persist_path* if it exists,
    otherwise create a new (empty) in-memory index that can be saved later.

    Call ``vector_store.save_local(persist_path)`` after adding documents
    to write the index to disk.
    """
    embeddings = get_embeddings()

    if os.path.isdir(persist_path) and os.listdir(persist_path):
        print(f"[FAISS] Loading existing index from '{persist_path}' …")
        vector_store = FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True,   # required by LangChain ≥ 0.2
        )
    else:
        print("[FAISS] No existing index found – creating a new one …")
        embedding_dim = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    return vector_store


def save_faiss_vector_store(vector_store: FAISS, persist_path: str = FAISS_INDEX_PATH) -> None:
    """Persist the FAISS index to *persist_path* (two files: .faiss + .pkl)."""
    os.makedirs(persist_path, exist_ok=True)
    vector_store.save_local(persist_path)
    print(f"[FAISS] Index saved to '{persist_path}'")



# ---------------------------------------------------------------------------
# Document loading (shared)
# ---------------------------------------------------------------------------
def load_docs(vector_store, force_reload: bool = False) -> None:
    """
    Fetch the blog post, chunk it, and add the chunks to *vector_store*.

    If the store already contains documents and *force_reload* is False the
    function returns early to avoid duplicate entries.
    """
    # Quick check: skip if documents are already present
    if not force_reload:
        try:
            existing = vector_store.similarity_search("test", k=1)
            if existing:
                print("[Loader] Documents already present in the store – skipping reload.")
                return
        except Exception:
            pass  # store is empty or doesn't support the check yet

    print(f"[Loader] Fetching and indexing '{BLOG_URL}' …")
    loader = WebBaseLoader(
        web_paths=(BLOG_URL,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    print(f"[Loader] Indexed {len(all_splits)} chunks.")


# ---------------------------------------------------------------------------
# Retrieval tool (closure so it captures the active vector_store)
# ---------------------------------------------------------------------------
def make_retrieve_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG demo with FAISS vector store")
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Re-fetch and re-index documents even if the store already has data.",
    )
    args = parser.parse_args()

    # ---- initialise the store ----
    vector_store = init_faiss_vector_store()

    # ---- load documents (skipped if already indexed) ----
    load_docs(vector_store, force_reload=args.force_reload)

    # ---- persist FAISS to disk after indexing ----
    save_faiss_vector_store(vector_store)

    # ---- build agent ----
    model = ChatOllama(model="llama3.1:8b", temperature=0.2, max_tokens=1000)
    tools = [make_retrieve_tool(vector_store)]
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    agent = create_agent(model, tools, system_prompt=prompt, checkpointer=InMemorySaver())

    # ---- chat loop ----
    while True:
        query = input("👤 You: ")
        if query.lower() == "quit":
            print("Agent: Goodbye!")
            break
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}})
        print("Agent:", response["messages"][-1].content)
