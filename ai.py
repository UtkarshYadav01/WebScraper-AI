from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


llm = ChatOllama(
    model="llama3",
    temperature=0
)

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def build_vector_store(text: str):
    """
    Split text into chunks and store embeddings in FAISS.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def answer_question(context: str, question: str) -> str:
    """
    Answers a question using FAISS-based retrieval.
    """

    vector_store = build_vector_store(context)

    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return "❌ Answer not found in the provided page content."

    combined_context = "\n\n".join(doc.page_content for doc in docs)

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
