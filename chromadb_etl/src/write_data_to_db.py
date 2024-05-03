from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama
import uuid
import PyPDF2
import os
import shutil

# docker mode: host.docker.internal
# local mode: 127.0.0.1

CHROMA_PATH = "chroma"
# normally: data, when test: ../data
DATA_PATH = "../data"
AGENT_MODEL = os.getenv("AGENT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def main():
    convert_all_pdfs_to_txt(DATA_PATH)
    generate_data_store()

def get_files_in_directory(directory):
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        files.append(file_path)
    return files

def check_folder_exists(folder_path):
    if os.path.exists(folder_path):
        print(f"The folder '{folder_path}' exists.")
    else:
        print(f"The folder '{folder_path}' does not exist.")

def check_file_exists(file_path):
    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
    else:
        print(f"The file '{file_path}' does not exist.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def convert_pdf_to_txt(input_file, output_file):
    with open(input_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        with open(output_file, 'w') as output:
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                output.write(text)

def convert_all_pdfs_to_txt(folder_path):
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    for pdf_file in pdf_files:
        print(str(pdf_file))
        pdf_path = os.path.join(folder_path, pdf_file)
        txt_file = os.path.splitext(pdf_file)[0] + ".txt"
        txt_path = os.path.join(folder_path, txt_file)
        convert_pdf_to_txt(pdf_path, txt_path)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[0]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    client = chromadb.HttpClient(host="host.docker.internal", port=8000, settings=Settings(allow_reset=True))
    client.reset()
    
    collection = client.create_collection(name="resume_collection", metadata={"hnsw:space": "cosine"})

    ollama_client = ollama.Client(host="http://host.docker.internal:11434")

    # check if model is already pulled
    models = [model['name'].replace(":latest", "") for model in ollama_client.list()['models']]

    if AGENT_MODEL not in models:
        print("model does not exist, pulling from ollama")
        ollama_client.pull(model=AGENT_MODEL)

    if EMBEDDING_MODEL not in models:
        print("embedding model does not exist, pulling from ollama")
        ollama_client.pull(model=EMBEDDING_MODEL)
    
    for chunk in chunks:
        # embed using ollama
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=chunk.page_content, keep_alive="-1m")
        embedding = response["embedding"]

        collection.add(
            ids=[str(uuid.uuid4())], metadatas=chunk.metadata, embeddings=[embedding], documents=chunk.page_content 
        )

    print(f"Saved {len(chunks)} chunks to chromadb instance.")

if __name__ == "__main__":
    main()