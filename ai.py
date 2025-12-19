from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import AZURE_AI_ENDPOINT_LLM
from config import DEPLOYMENT_NAME_LLM


llm = ChatOpenAI(
    model_name=DEPLOYMENT_NAME_LLM,
    base_url=AZURE_AI_ENDPOINT_LLM,
    temperature=0
)


def answer_question(context: str, question: str) -> str:
    """
    Answers a question strictly based on the given context.
    If answer is not present in context, says so clearly.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant.

Answer the question ONLY using the information provided in the context below.
If the answer is not present in the context, reply exactly:
"❌ Answer not found in the provided page content."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    response = llm.invoke(
        prompt.format(
            context=context[:6000],  # limit context size
            question=question
        )
    )

    return response.content
