import os, shutil
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

DATA_PATH = "rules"
DB_PATH = "chroma"
load_dotenv()

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def populate_database(chunks):
    reset_database()

    db = Chroma. from_documents(
        chunks,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory = DB_PATH
    )
    db.persist()

def reset_database():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

def setup():
    documents = load_documents()
    chunks = split_documents(documents)
    populate_database(chunks)
