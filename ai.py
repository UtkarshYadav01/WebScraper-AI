from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY


# Initialize LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    temperature=0.3
)


def summarize_text(text: str) -> str:
    """
    Summarizes scraped web content using LLM.
    """

    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
        You are an AI assistant.
        Summarize the following web content in simple and clear bullet points.

        Content:
        {content}
        """
    )

    response = llm.invoke(
        prompt.format(content=text[:6000])  # limit input size
    )

    return response.content
