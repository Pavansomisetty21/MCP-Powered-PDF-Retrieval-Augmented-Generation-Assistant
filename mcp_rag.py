import os
import requests
import pdfplumber
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS as LangchainFAISS


load_dotenv()
mcp = FastMCP("MCP Powered RAG Server")


vectorstore = None
doc_objects = []


embedder = SpacyEmbeddings(model_name="en_core_web_sm")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=23)



def _extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_embeddings(pdf_path):
    global vectorstore, doc_objects

    text = _extract_text_from_pdf(pdf_path)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    doc_objects.extend(docs)

    if vectorstore is None:
      
        vectorstore = LangchainFAISS.from_documents(docs, embedder)
    else:
      
        vectorstore.add_documents(docs)

    print(f"[info] Indexed {len(docs)} chunks from PDF.")

custom_prompt = PromptTemplate.from_template(
    """You are a AI Assistant provides answer to the user question and follow the instructions. 
Context: {context} 
Question: {question} 
History: {chat_history} 
."""
)

def llm():
    api_key=os.environ.get("GROQ_API_KEY")
    model="llama-3.3-70b-versatile"
    return ChatGroq(api_key=api_key,model=model)

llm = llm()

def _create_chain():
    if vectorstore is None:
        raise ValueError("No documents indexed. Please index a PDF file first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True),
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        output_key="answer"
    )

@mcp.tool()
def rag_query(question: str) -> str:
    print("[debug] RAG Query:", question)
    chain = _create_chain()
    result = chain.invoke({"question": question})
    response = result["answer"]

    if not response:
        result = chain.invoke({"question": question})
        response = result["answer"]

    return response

@mcp.tool()
def index_pdf_file(file_path: str) -> str:
    print(f"[info] Indexing file: {file_path}")
    get_embeddings(file_path)
    return "true"


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_files")
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.endswith(".pdf"):
                get_embeddings(os.path.join(sample_dir, file))

    mcp.run(transport="sse")
