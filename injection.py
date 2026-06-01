import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf():
    loader=DirectoryLoader(
        path="data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs=loader.load()
    print(len(docs))
    return docs

 # Step 2: Create Chunks
def create_chunks(docs):
    text_split=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_split.split_documents(docs)
    print(len(chunks))
    return chunks

#emdeddings
def create_embeddings(chunks):
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_faiss_path="vectorstore/db_faiss"
    vectorstore=FAISS.from_documents(chunks,embedding)
    vectorstore.save_local(db_faiss_path)
    return vectorstore
    
# Main
if __name__ == "__main__":
    docs=load_pdf()
    chunks=create_chunks(docs)
    vectorstore=create_embeddings(chunks)