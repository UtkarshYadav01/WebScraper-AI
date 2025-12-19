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


def answer_question(vector_store: FAISS, question: str):
    """
    Returns:
    - answer (str)
    - sources (list[str])
    """

    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return (
            "❌ Answer not found in the provided page content.",
            []
        )

    context = "\n\n".join(doc.page_content for doc in docs)

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
        prompt.format(context=context, question=question)
    )

    sources = [doc.page_content for doc in docs]

    return response.content, sources
