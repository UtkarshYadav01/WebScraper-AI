import streamlit as st

from scraper import scrape_website
from ai import answer_question


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.write("🌐 Ask questions based only on the content of a web page.")

url = st.text_input("🔗 Website URL")

if "page_content" not in st.session_state:
    st.session_state.page_content = None

if st.button("🕷️ Load Page Content"):
    if not url:
        st.warning("⚠️ Please enter a valid URL")
    else:
        with st.spinner("🕷️ Scraping website content..."):
            try:
                st.session_state.page_content = scrape_website(url)
                print("✅ Page content loaded successfully")
                st.success("📄 Page content loaded. You can now ask questions.")
            except Exception as e:
                print("❌ Error during scraping:", e)
                st.error("❌ Failed to load page content. Check console.")
                st.stop()

if st.session_state.page_content:
    question = st.text_input("❓ Ask a question based on this page")

    if st.button("🤖 Get Answer"):
        if not question:
            st.warning("⚠️ Please enter a question")
        else:
            with st.spinner("🤖 Finding answer from page content..."):
                try:
                    answer = answer_question(
                        st.session_state.page_content,
                        question
                    )
                    print("✅ Question answered successfully")
                except Exception as e:
                    print("❌ Error during Q&A:", e)
                    st.error("❌ Failed to answer question. Check console.")
                    st.stop()

            st.subheader("📌 Answer")
            st.write(answer)
