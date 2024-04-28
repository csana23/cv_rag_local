from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import PyPDF2
import os
import shutil

# CHROMA_PATH = os.path.join(os.getcwd(), "../../chroma")
# DATA_PATH = os.path.join(os.getcwd(), "../../data")

CHROMA_PATH = "chroma"
DATA_PATH = "../data"
# SRC_PATH = "app/src"
# FILE_PATH = "write_data_to_db.py"

def main():
    convert_all_pdfs_to_txt(DATA_PATH)
    chunks = split_text(load_documents())

    print(chunks)
    # generate_data_store()
    # check_folder_exists(SRC_PATH)
    # print(os.getcwd())
    # print(get_files_in_directory(os.getcwd()))

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
        chunk_size=1000,
        chunk_overlap=100,
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
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()