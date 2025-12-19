import streamlit as st

from scraper import scrape_website
from ai import summarize_text


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.write("🌐 Enter a website URL to scrape and summarize using AI.")

url = st.text_input("🔗 Website URL")

if st.button("🚀 Scrape & Summarize"):
    if not url:
        st.warning("⚠️ Please enter a valid URL")
    else:
        with st.spinner("🕷️ Scraping website content..."):
            try:
                content = scrape_website(url)
                print("✅ Scraping completed successfully")
            except Exception as e:
                print("❌ Error during scraping:", e)  # console log
                st.error("❌ Failed to scrape the website. Check console for details.")
                st.stop()

        with st.spinner("🤖 Generating AI summary..."):
            try:
                summary = summarize_text(content)
                print("✅ AI summary generated successfully")
            except Exception as e:
                print("❌ Error during AI summarization:", e)  # console log
                st.error("❌ Failed to generate AI summary. Check console for details.")
                st.stop()

        st.success("🎉 Done successfully!")
        st.subheader("📌 AI Summary")
        st.write(summary)
