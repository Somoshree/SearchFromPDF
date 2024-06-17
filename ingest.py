from langchain.document_loaders import PyPDFLoader,DirectoryLoader,PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory= "db"

def main():
    for root,dir,files in os.walk('docs'):
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                loader = PDFMinerLoader(os.path.join(root,file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    local_embedding_model = SentenceTransformer("/model/all-MiniLM-L6-v2")
    embeddings =  SentenceTransformerEmbeddings(model=local_embedding_model)
    db = Chroma.from_documents(texts,embeddings,persist_directory=persist_directory)
    db.persist()
    db=None


if __name__ == "__main__":
    main()
