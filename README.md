import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Web RAG Chatbot", layout="wide")
st.title("Web RAG Chatbot")

# Session state init
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show processed URL + clear chat button
col1, col2 = st.columns([4, 1])
with col1:
    if st.session_state.processed_url:
        st.success(f"Loaded: {st.session_state.processed_url}")
    else:
        st.info(" Enter a URL and click 'Process Website' to start")
with col2:
    if st.session_state.messages:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# URL input + process button
website_url = st.text_input("Website URL", placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence")
process_website = st.button("Process Website", type="primary", use_container_width=True)

@st.cache_resource(show_spinner=False)
def create_vector_store(url: str) -> FAISS:
    loader = WebBaseLoader(url)
    documents = loader.load()
    if not documents or not documents[0].page_content.strip():
        raise ValueError("No content extracted. Site may use JavaScript rendering.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(split_docs, embeddings)

def format_docs(docs):
    return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])

def get_rag_chain():
    """Returns a streaming RAG chain"""
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for {url}.
Use chat history to understand follow-up questions.
Answer ONLY from the provided context. If not in context, say "I don't know based on the webpage."
Keep answers concise.

Chat History:
{chat_history}

Context from webpage:
{context}

Question: {question}
""")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        streaming=True # Enable streaming
    )

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "url": lambda x: x["url"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# API key check
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to.env file.")
    st.stop()

# Process website
if process_website:
    if not website_url:
        st.error("Please enter a website URL.")
    else:
        try:
            with st.spinner("Loading website and building index..."):
                st.session_state.vectorstore = create_vector_store(website_url)
                st.session_state.processed_url = website_url
                st.session_state.messages = [] # Reset chat on new URL
            st.success(f"Processed! Found {st.session_state.vectorstore.index.ntotal} chunks.")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Show sources"):
                for i, doc in enumerate(message["sources"], 1):
                    st.caption(f"**Chunk {i}** - {doc.metadata.get('source', 'Unknown')}")
                    st.text(doc.page_content[:400] + "...")

# Chat input - only show if website is processed
if st.session_state.vectorstore:
    if prompt := st.chat_input("Ask something about the webpage"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response with streaming
        with st.chat_message("assistant"):
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
            rag_chain, retriever = get_rag_chain()

            # Stream the response
            response = st.write_stream(rag_chain.stream({
                "question": prompt,
                "chat_history": history_str,
                "url": st.session_state.processed_url
            }))

            # Get sources for display
            sources = retriever.invoke(prompt)

        # Save assistant message + sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
else:
    st.warning("Process a website first to start chatting.")
    # Web-RAG-Chatbot
