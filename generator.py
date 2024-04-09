import os
from langchain_community.llms import HuggingFaceEndpoint
from pydantic import ValidationError

secret_token = os.getenv("HUGGINGFACE_API_TOKEN")

def load_llm(repo_id="mistralai/Mistral-7B-Instruct-v0.2"):
    '''
    Load the LLM from the HuggingFace model hub
    Args:
        repo_id (str): The HuggingFace model ID
    Returns:
        llm (HuggingFaceEndpoint): The LLM model
    '''

    try:
        repo_id = repo_id
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            model_kwargs={"max_length": 128}, temperature=0.2, huggingfacehub_api_token = secret_token
        )
        return llm
    except ValidationError as e:
        print("Validation Error:", e)
        # Log or handle the validation error appropriately
        return None
    except Exception as e:
        print("Error:", e)
        # Log or handle other exceptions
        return None


