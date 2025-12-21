import streamlit as st

from graph import build_rag_graph


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.write("🌐 Smart RAG with LangGraph (conditional routing).")

url = st.text_input("🔗 Website URL")
question = st.text_input("❓ Ask a question based on this page")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if st.button("🚀 Run Agent"):
    if not question:
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("🧠 Running LangGraph agent..."):
            try:
                graph = build_rag_graph()

                result = graph.invoke({
                    "url": url,
                    "question": question,
                    "page_content": None,
                    "vector_store": st.session_state.vector_store,
                    "answer": None,
                    "sources": None
                })

                # Persist index if created
                if result.get("vector_store"):
                    st.session_state.vector_store = result["vector_store"]

                print("✅ Agent executed successfully")

            except Exception as e:
                print("❌ Agent execution error:", e)
                st.error("❌ Failed to run agent. Check console.")
                st.stop()

        st.subheader("📌 Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("📚 Sources")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"Source {i}"):
                    st.write(source)
