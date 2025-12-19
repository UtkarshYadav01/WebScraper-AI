from dotenv import load_dotenv
import os

load_dotenv()

AZURE_AI_ENDPOINT_LLM = os.getenv("AZURE_AI_ENDPOINT_LLM")
DEPLOYMENT_NAME_LLM = os.getenv("DEPLOYMENT_NAME_LLM")