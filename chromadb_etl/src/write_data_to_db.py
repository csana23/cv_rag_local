from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import PyPDF2
import os
import shutil

CHROMA_PATH = "../../chroma"
DATA_PATH = "../../data/"


def main():
    convert_all_pdfs_to_txt(DATA_PATH)
    generate_data_store()

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
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
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