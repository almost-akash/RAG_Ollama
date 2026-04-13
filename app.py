import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
import tempfile

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("Chat with your PDF (Local LLM)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("Upload a PDF",type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.file.write(uploaded_file.read())
            file_path = tmp.name

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        db = FAISS.from_documents(chunks,embeddings)

        llm = ChatOllama(model="llama3", temperature=0)

        retriever = db.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm= llm,
            retriever=retriever
        )

        st.session_state.qa_chain = qa_chain

    st.success("Document processed! You can now ask questions.")

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

query = st.text_input("Ask a question about your document: ")

if query and st.session_state.qa_chain:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})

        st.subheader("Answer")
        st.write(result["result"])

        docs = retriever.get_relevant_documents(query)

        with st.expander("Sources"):
            for doc in docs:
                st.write(doc.page_content[:300])
                st.markdown("---")

elif query:
    st.warning("Please upload a document first.")