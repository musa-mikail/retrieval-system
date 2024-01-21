from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def document_loader(doc_path):
    loader = PyMuPDFLoader(doc_path)
    docs = loader.load()
    return docs

def splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100,
    )
    chunks =  splitter.split_documents(docs)
    return chunks

def vector_store(chunks, embed_model = "models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db
    
def prompter():

    template = """Use the following pieces of context to answer the question at the end.
    Use three sentences maximum and keep the answer as concise and smart as possible.
    Always say "Thank You" at the end of the answer.

    {context}

    Query: {query}

    Helpful Answer:"""

    return PromptTemplate.from_template(template)

def LLM(llm_model="gemini-pro"):
    print("Initializing the Language model...")
    return ChatGoogleGenerativeAI(model=llm_model)
