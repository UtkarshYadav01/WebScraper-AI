from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from scraper import scrape_website
from ai import build_vector_store, answer_question


# -------- State Definition --------

class GraphState(TypedDict):
    url: str
    question: str
    page_content: Optional[str]
    vector_store: Optional[object]
    answer: Optional[str]
    sources: Optional[list]


# -------- Node Functions --------

def scrape_node(state: GraphState):
    print("🕷️ [Graph] Scraping page...")
    content = scrape_website(state["url"])
    return {"page_content": content}


def index_node(state: GraphState):
    print("📦 [Graph] Building FAISS index...")
    vector_store = build_vector_store(state["page_content"])
    return {"vector_store": vector_store}


def qa_node(state: GraphState):
    print("🤖 [Graph] Answering question using RAG...")
    answer, sources = answer_question(
        state["vector_store"],
        state["question"]
    )
    return {
        "answer": answer,
        "sources": sources
    }


# -------- Graph Builder --------

def build_rag_graph():
    graph = StateGraph(GraphState)

    graph.add_node("scrape", scrape_node)
    graph.add_node("index", index_node)
    graph.add_node("qa", qa_node)

    graph.set_entry_point("scrape")

    graph.add_edge("scrape", "index")
    graph.add_edge("index", "qa")
    graph.add_edge("qa", END)

    return graph.compile()
