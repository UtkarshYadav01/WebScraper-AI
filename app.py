import streamlit as st

from scraper import scrape_website
from ai import summarize_text


st.set_page_config(page_title="WebScraper AI", layout="wide")

st.title("🕷️ WebScraper AI")
st.write("Enter a website URL to scrape and summarize using AI.")

url = st.text_input("Website URL")

if st.button("Scrape & Summarize"):
    if not url:
        st.warning("Please enter a valid URL")
    else:
        with st.spinner("Scraping website..."):
            try:
                content = scrape_website(url)
            except Exception as e:
                st.error(f"Error while scraping: {e}")
                st.stop()

        with st.spinner("Generating AI summary..."):
            try:
                summary = summarize_text(content)
            except Exception as e:
                st.error(f"Error while generating summary: {e}")
                st.stop()

        st.success("Done!")
        st.subheader("📌 AI Summary")
        st.write(summary)
