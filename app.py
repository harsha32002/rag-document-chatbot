import os
import shutil
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()

st.set_page_config(page_title="RAG Document Chatbot")

st.title("RAG Document Q&A Chatbot")
st.write("Upload a PDF document and ask questions from it.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    st.write("File name:", uploaded_file.name)

    pdf_reader = PdfReader(uploaded_file)

    documents = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number
                    }
                )
            )

    st.write("Total pages extracted:", len(documents))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
)

    chunks = text_splitter.split_documents(documents)

    st.write("Total chunks created:", len(chunks))

    if st.button("Create Vector Database"):
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        st.success("Vector database created successfully!")

st.divider()

st.subheader("Ask a question from the document")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        results = vector_store.similarity_search(question, k=5)

        context = ""

        for doc in results:
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "Unknown page")
            context += f"Source: {source}, Page: {page}\n"
            context += doc.page_content + "\n\n"

        prompt = f"""
You are a helpful assistant. Answer the question only using the context below.
If the answer is not present in the context, say: "I could not find this information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)
        if "I could not find this information in the uploaded document" not in response.content:
            st.subheader("Sources")
            for i, doc in enumerate(results, start=1):
                st.write(
                    f"{i}. {doc.metadata.get('source', 'Unknown source')} "
                    f"- Page {doc.metadata.get('page', 'Unknown page')}"
                    )
        else:
            st.info("No relevant source found in the uploaded document.")

        with st.expander("Retrieved chunks used"):
            for i, doc in enumerate(results, start=1):
                st.write(f"### Chunk {i}")
                st.write(
                    f"Source: {doc.metadata.get('source', 'Unknown source')}, "
                    f"Page: {doc.metadata.get('page', 'Unknown page')}"
                )
                st.write(doc.page_content)
    else:
        st.warning("Please enter a question first.")