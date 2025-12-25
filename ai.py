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
    - sources (list[str]) -> ONLY chunks actually used
    """

    # Retrieve more than needed (upper bound)
    retrieved_docs = vector_store.similarity_search(question, k=15)

    if not retrieved_docs:
        return "❌ Answer not found in the provided page content.", []

    # Prepare numbered context so LLM can reference chunks
    numbered_context = []
    for idx, doc in enumerate(retrieved_docs, 1):
        numbered_context.append(f"[{idx}] {doc.page_content}")

    context_text = "\n\n".join(numbered_context)

    memory_text = ""
    if memory:
        memory_text = "\n".join(
            f"Q: {m['question']}\nA: {m['answer']}"
            for m in memory[-3:]
        )

    prompt = PromptTemplate(
        input_variables=["context", "memory", "question"],
        template="""
You are an AI assistant with strict rules.

RULES:
- Answer ONLY using the context below
- Select ONLY the chunks that were ACTUALLY used to answer
- If no chunk is relevant, say:
  "❌ Answer not found in the provided page content."
- Return used chunk numbers as a comma-separated list

Conversation Memory:
{memory}

Context:
{context}

Question:
{question}

Respond in EXACTLY this format:

ANSWER:
<answer here>

SOURCES:
<comma separated chunk numbers OR empty>
"""
    )

    response = llm.invoke(
        prompt.format(
            context=context_text,
            memory=memory_text,
            question=question
        )
    )

    raw = response.content.strip()

    # ---- Parse LLM response safely ----
    if "SOURCES:" not in raw:
        return raw, []

    answer_part, sources_part = raw.split("SOURCES:", 1)
    answer = answer_part.replace("ANSWER:", "").strip()

    source_ids = [
        int(x.strip())
        for x in sources_part.strip().split(",")
        if x.strip().isdigit()
    ]

    used_sources = [
        retrieved_docs[i - 1].page_content
        for i in source_ids
        if 0 < i <= len(retrieved_docs)
    ]

    return answer, used_sources
