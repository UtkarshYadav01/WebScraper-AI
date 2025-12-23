import streamlit as st

from graph import build_rag_graph


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.caption("Chat with a web page using RAG + LangGraph")

# -------------------------------
# Session State Initialization
# -------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat" not in st.session_state:
    st.session_state.chat = []
    print("🆕 Initialized new chat session")

if "url" not in st.session_state:
    st.session_state.url = ""


# -----------------------------
# Sidebar: Configuration Panel
# -----------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    url = st.text_input(
        "🔗 Website URL",
        value=st.session_state.url or ""
    )

    if st.button("🕷️ Scrape & Index"):
        if not url:
            st.warning("⚠️ Please enter a valid URL")
        else:
            with st.spinner("Scraping and indexing page..."):
                try:
                    graph = build_rag_graph()

                    result = graph.invoke({
                        "url": url,
                        "question": "initialization",
                        "page_content": None,
                        "vector_store": None,
                        "answer": None,
                        "sources": None
                    })

                    st.session_state.vector_store = result.get("vector_store")
                    st.session_state.url = url
                    st.session_state.chat = []  # reset chat

                    print("✅ Page scraped & indexed")
                    st.success("✅ Page indexed successfully")

                except Exception as e:
                    print("❌ Indexing error:", e)
                    st.error("❌ Failed to index page. Check console.")

    if st.session_state.vector_store:
        st.success("📦 Page is indexed and ready")
    else:
        st.info("ℹ️ No page indexed yet")

# -------------------------------
# Chat Section
# -------------------------------

st.divider()
st.subheader("💬 Chat")

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources only for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("📚 **Sources**")
            for i, source in enumerate(msg["sources"], 1):
                with st.expander(f"Source {i}"):
                    st.write(source)


# Chat input
if user_prompt := st.chat_input("Ask something based on the page..."):
    if not st.session_state.vector_store:
        st.warning("⚠️ Please scrape a page first using Configuration.")
    else:
        # Add user message
        st.session_state.chat.append({
            "role": "user",
            "content": user_prompt
        })

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    graph = build_rag_graph()

                    result = graph.invoke({
                        "url": st.session_state.url,
                        "question": user_prompt,
                        "page_content": None,
                        "vector_store": st.session_state.vector_store,
                        "answer": None,
                        "sources": None
                    })

                    answer = result["answer"]
                    sources = result.get("sources", [])

                    # Render answer immediately
                    st.markdown(answer)

                    # ✅ Render sources immediately (THIS FIXES THE GLITCH)
                    if sources:
                        st.markdown("📚 **Sources**")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}"):
                                st.write(source)

                except Exception as e:
                    print("❌ Agent execution error:", e)
                    answer = "❌ Something went wrong. Check console."
                    sources = []
                    st.error(answer)

        # Save assistant message
        st.session_state.chat.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
