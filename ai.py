from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Local LLM (Ollama)
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# Local embeddings (Ollama)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def build_vector_store(text: str) -> FAISS:
    """
    Build FAISS vector store from scraped page content.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def answer_question(vector_store: FAISS, question: str) -> str:
    """
    Answer a question strictly using retrieved page content.
    """

    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return "❌ Answer not found in the provided page content."

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
        prompt.format(
            context=context,
            question=question
        )
    )

    return response.content
