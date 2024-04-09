from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os

# Function to create embeddings
# def create_embeddings(text_chunks):
#     embeddings = embeddings_model.encode(text_chunks, show_progress_bar=True)
#     return embeddings

curr_dir = os.getcwd()
db_path = 'chroma_db'

def process_pdf_document(file_path_list):
    '''
    Process a PDF document and return the documents and text splitters
    Args:
        file_path (str): The path to the PDF document
        parent_chunk_size (int): The size of the parent chunks
        child_chunk_size (int): The size of the child chunks
    Returns:
        documents (list): The list of documents
        parent_splitter (RecursiveCharacterTextSplitter): The text splitter for the parent documents
        child_splitter (RecursiveCharacterTextSplitter): The text splitter for the child documents
    '''
    # # Load the PDF document
    # loader = PyMuPDFLoader(file_path)
    # documents = loader.load()

    loaders = [PyMuPDFLoader(file_path) for file_path in file_path_list]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents


# Function to create the vectorstore
def create_vectorstore(embeddings_model="all-MiniLM-L6-v2"):
    '''
    Create the vectorstore and store for the documents
    Args:
        embeddings_model (HuggingFaceEmbeddings): The embeddings model
        documents (list): The list of documents
    Returns:
        vectorstore (Chroma): The vectorstore
        store (InMemoryStore): The store
    '''

    # Initialize the embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # # This text splitter is used to create the parent documents
    # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    # # This text splitter is used to create the child documents
    # # It should create documents smaller than the parent
    # child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # The vectorstore to use to index the child chunks
    # vectorstore = Chroma(
    #     collection_name="split_parents", embedding_function=embeddings_model
    # )
    vectordb = Chroma(persist_directory=db_path,
                  embedding_function=embeddings_model)
    
    # The storage layer for the parent documents
    store = InMemoryStore()

    return vectordb, store



def rag_retriever(vectorstore, store, documents, parent_splitter, child_splitter):
    '''
    Create the retriever for the RAG model
    Args:
        vectorstore (Chroma): The vectorstore
        store (InMemoryStore): The store
        parent_splitter (RecursiveCharacterTextSplitter): The text splitter for the parent documents
        child_splitter (RecursiveCharacterTextSplitter): The text splitter for the child documents
    Returns:
        retriever (ParentDocumentRetriever): The retriever
        
    '''

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        docs=documents
    )

    # retriever.add_documents(documents)
    # retriever = vectorstore.as_retriever()

    return retriever