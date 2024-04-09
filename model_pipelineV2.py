# Implement Classification

import os
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from generator import load_llm
from langchain.prompts import PromptTemplate
from retrieverV2 import process_pdf_document, create_vectorstore, rag_retriever
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import pickle

class VectorStoreSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = create_vectorstore()  # Your existing function to create the vectorstore
        return cls._instance

class LanguageModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = load_llm()  # Your existing function to load the LLM
        return cls._instance


class ModelPipeLine:
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    def __init__(self):
        self.curr_dir = os.path.dirname(__file__)
        self.knowledge_dir = 'knowledge'
        self.prompt_dir = 'prompts'
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        self._documents = None  # Initialize as None for lazy loading
        self.vectorstore, self.store = VectorStoreSingleton.get_instance()
        self._retriever = None  # Corrected: Initialize _retriever as None for lazy loading
        self.llm = LanguageModelSingleton.get_instance()
        self.memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

    @property
    def documents(self):
        if self._documents is None:
            self._documents = process_pdf_document([
                os.path.join(self.knowledge_dir, 'depression_1.pdf'),
                os.path.join(self.knowledge_dir, 'depression_2.pdf')
            ])
        return self._documents

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = rag_retriever(self.vectorstore, self.store, self.documents, self.parent_splitter, self.child_splitter)
        return self._retriever
    
    def get_prompts(self, system_file_path='system_prompt_template.txt', 
                    condense_file_path='condense_question_prompt_template.txt'):
        
        with open(os.path.join(self.prompt_dir, system_file_path), 'r') as f:
            system_prompt_template = f.read()

        with open(os.path.join(self.prompt_dir, condense_file_path), 'r') as f:
            condense_question_prompt = f.read()  

        # create message templates
        ANSWER_PROMPT = ChatPromptTemplate.from_template(system_prompt_template)

        # create message templates
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt)

        return ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT
    

    def _combine_documents(self,docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
       
    def create_final_chain(self):

        answer_prompt, condense_question_prompt = self.get_prompts()
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )
        # Now we calculate the standalone question
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | condense_question_prompt
            | self.llm,
        }
        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | self.retriever,
            "question": lambda x: x["standalone_question"],
        }
        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: self._combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | answer_prompt | self.llm,
            "docs": itemgetter("docs"),
        }
        # And now we put it all together!
        final_chain = loaded_memory | standalone_question | retrieved_documents | answer

        return final_chain
    

    def call_conversational_rag(self,question, chain):
        """
        Calls a conversational RAG (Retrieval-Augmented Generation) model to generate an answer to a given question.
        This function sends a question to the RAG model, retrieves the answer, and stores the question-answer pair in memory 
        for context in future interactions.
        Parameters:
        question (str): The question to be answered by the RAG model.
        chain (LangChain object): An instance of LangChain which encapsulates the RAG model and its functionality.
        memory (Memory object): An object used for storing the context of the conversation.
        Returns:
        dict: A dictionary containing the generated answer from the RAG model.
        """
        
        # Prepare the input for the RAG model
        inputs = {"question": question}

        # Invoke the RAG model to get an answer
        result = chain.invoke(inputs)
        
        # Save the current question and its answer to memory for future context
        self.memory.save_context(inputs, {"answer": result["answer"]})
        
        # Return the result
        return result

    def process_message(self, message, lower_case=True, stem=True, stop_words=True):
        if lower_case:
            message = message.lower()
        
        words = word_tokenize(message)
        
        if stop_words:
            sw = set(stopwords.words('english'))
            words = [word for word in words if word not in sw]
        
        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        return ' '.join(words)

    def load_model(self):
        model_path = 'sentiment_classifier.pkl'
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    
    def predict_classification(self, message):
        s_model = self.load_model()
        processed_msg = self.process_message(message)
        pred_label = s_model.predict([processed_msg])
        return pred_label[0]

#ml_pipeline = ModelPipeLine()
#final_chain = ml_pipeline.create_final_chain()
#question = "i am feeling sad"
#res = ml_pipeline.call_conversational_rag(question,final_chain)
#print(res['answer'])
