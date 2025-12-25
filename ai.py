from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


llm = ChatOllama(
    model="llama3",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def build_vector_store(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)


def answer_question(vector_store: FAISS, question: str, memory: list):
    """
    Returns:
    - answer (str)
    - sources (list[str])
    """

    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return "❌ Answer not found in the provided page content.", []

    context = "\n\n".join(doc.page_content for doc in docs)

    memory_text = ""
    if memory:
        memory_text = "\n".join(
            f"Q: {m['question']}\nA: {m['answer']}"
            for m in memory[-3:]  # last 3 turns
        )

    prompt = PromptTemplate(
        input_variables=["context", "memory", "question"],
        template="""
You are an AI assistant.

You MUST follow these rules:
- Answer ONLY using the provided context
- Use conversation memory ONLY for continuity
- If answer is not in context, say:
  "❌ Answer not found in the provided page content."

Conversation Memory:
{memory}

Context:
{context}

Question:
{question}

Answer:
"""
    )

    response = llm.invoke(
        prompt.format(context=context, memory=memory_text, question=question)
    )

    sources = [doc.page_content for doc in docs]

    return response.content, sources
