from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from scraper import scrape_website
from ai import build_vector_store, answer_question


# -------- State Definition --------

class GraphState(TypedDict):
    url: Optional[str]
    question: str
    page_content: Optional[str]
    vector_store: Optional[object]
    answer: Optional[str]
    sources: Optional[list]


# -------- Router Decision Function --------

def route_decision(state: GraphState) -> str:
    """
    Decide next step based on current state.
    """
    print("🧭 [Graph] Routing decision...")

    if not state.get("url"):
        return "error"

    if state.get("vector_store"):
        return "qa"

    return "scrape"


# -------- Node Functions --------

def route_node(state: GraphState):
    # Router node MUST return dict
    return {}


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


def error_node(state: GraphState):
    print("❌ [Graph] Invalid input state")
    return {
        "answer": "❌ Please provide a valid URL before asking questions.",
        "sources": []
    }


# -------- Graph Builder --------

def build_rag_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("index", index_node)
    graph.add_node("qa", qa_node)
    graph.add_node("error", error_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "scrape": "scrape",
            "qa": "qa",
            "error": "error",
        }
    )

    graph.add_edge("scrape", "index")
    graph.add_edge("index", "qa")
    graph.add_edge("qa", END)
    graph.add_edge("error", END)

    return graph.compile()
