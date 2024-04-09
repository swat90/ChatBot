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
db_path = 'chroma_db_v2'

class QuestionRetriever:
   
    def load_documents(self,file_name):
      current_directory = os.getcwd()
      data_directory = os.path.join(current_directory, "data")
      file_path = os.path.join(data_directory, file_name)
      loader = TextLoader(file_path)
      documents = loader.load()
      return documents
    
    def store_data_in_vector_db(self,documents):
    #   global db
      text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0,separator="\n")
      docs = text_splitter.split_documents(documents)
      # create the open-source embedding function
      embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
      # print(docs)
      # load it into Chroma
      db = Chroma.from_documents(docs, embedding_function)
      return db

    def get_response(self, user_query):
        db=self.store_data_in_vector_db(documents)

        docs = db.similarity_search(user_query)
        most_similar_question = docs[0].page_content.split("\n")[0]  # Extract the first question
        if user_query==most_similar_question:
          most_similar_question=docs[1].page_content.split("\n")[0]

        print(most_similar_question)
        return most_similar_question
        
def process_pdf_document(file_path, parent_chunk_size=2000, child_chunk_size=500):
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
    # Load the PDF document
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Initialize text splitters for parent and child documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)

    return documents, parent_splitter, child_splitter


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

    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="split_parents", embedding_function=embeddings_model
    )
    vectordb = Chroma(persist_directory=db_path,
                  embedding_function=embeddings_model)
    # The storage layer for the parent documents
    store = InMemoryStore()

    return vectordb, store



def rag_retriever(vectorstore):
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

    # retriever = ParentDocumentRetriever(
    #     vectorstore=vectorstore,
    #     docstore=store,
    #     child_splitter=None,
    #     parent_splitter=None,
    #     docs=documents
    # )

    # retriever.add_documents(documents)
    retriever = vectorstore.as_retriever()

    return retriever




# def retrieve_context(query, top_k):

#     # Retrieve the top k similar documents
#     sub_docs = vectorstore.similarity_search(query, k=top_k, return_documents=True)

#     # Get the context of the first document
#     context = sub_docs[0].page_content

#     return context
