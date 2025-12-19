import streamlit as st

from scraper import scrape_website
from ai import build_vector_store, answer_question


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.write("🌐 Ask questions strictly based on a web page’s content.")

url = st.text_input("🔗 Website URL")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if st.button("🕷️ Load Page Content"):
    if not url:
        st.warning("⚠️ Please enter a valid URL")
    else:
        with st.spinner("🕷️ Scraping and indexing page..."):
            try:
                content = scrape_website(url)
                st.session_state.vector_store = build_vector_store(content)
                print("✅ Page scraped and indexed successfully")
                st.success("📄 Page indexed. You can now ask questions.")
            except Exception as e:
                print("❌ Error during indexing:", e)
                st.error("❌ Failed to load page. Check console.")
                st.stop()

if st.session_state.vector_store:
    question = st.text_input("❓ Ask a question based on this page")

    if st.button("🤖 Get Answer"):
        if not question:
            st.warning("⚠️ Please enter a question")
        else:
            with st.spinner("🤖 Searching page content..."):
                try:
                    answer, sources = answer_question(
                        st.session_state.vector_store,
                        question
                    )
                    print("✅ Answer generated with citations")
                except Exception as e:
                    print("❌ Error during Q&A:", e)
                    st.error("❌ Failed to answer question. Check console.")
                    st.stop()

            st.subheader("📌 Answer")
            st.write(answer)

            if sources:
                st.subheader("📚 Sources")
                for i, source in enumerate(sources, 1):
                    with st.expander(f"Source {i}"):
                        st.write(source)
