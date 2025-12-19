from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import AZURE_AI_ENDPOINT_LLM
from config import DEPLOYMENT_NAME_LLM


llm = ChatOpenAI(
    model_name=DEPLOYMENT_NAME_LLM,
    base_url=AZURE_AI_ENDPOINT_LLM,
    temperature=0
)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def select_relevant_chunks(chunks, question, max_chunks=3):
    """
    Very simple relevance selection using keyword overlap.
    """
    question_words = set(question.lower().split())
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:max_chunks] if score > 0]


def answer_question(context: str, question: str) -> str:
    """
    Answers a question strictly using relevant chunks from context.
    """

    chunks = chunk_text(context)
    relevant_chunks = select_relevant_chunks(chunks, question)

    if not relevant_chunks:
        return "❌ Answer not found in the provided page content."

    combined_context = "\n\n".join(relevant_chunks)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant.

Answer the question ONLY using the context below.
If the answer is not present, reply exactly:
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
            context=combined_context,
            question=question
        )
    )

    return response.content
