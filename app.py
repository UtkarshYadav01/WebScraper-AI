import streamlit as st

from graph import build_rag_graph


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.write("🌐 Ask questions strictly based on a web page’s content.")

url = st.text_input("🔗 Website URL")
question = st.text_input("❓ Ask a question based on this page")

if st.button("🚀 Run RAG Workflow"):
    if not url or not question:
        st.warning("⚠️ Please enter both URL and question")
    else:
        with st.spinner("🧠 Running LangGraph RAG workflow..."):
            try:
                graph = build_rag_graph()

                result = graph.invoke({
                    "url": url,
                    "question": question,
                    "page_content": None,
                    "vector_store": None,
                    "answer": None,
                    "sources": None
                })

                print("✅ LangGraph workflow executed successfully")

            except Exception as e:
                print("❌ Graph execution error:", e)
                st.error("❌ Failed to execute workflow. Check console.")
                st.stop()

        st.subheader("📌 Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.subheader("📚 Sources")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"Source {i}"):
                    st.write(source)
