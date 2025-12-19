import requests
from bs4 import BeautifulSoup


def scrape_website(url: str) -> str:
    """
    Scrapes visible text content from a web page.
    Returns clean plain text.
    """

    # 1. Fetch page
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # 2. Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 3. Remove unwanted tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # 4. Extract text
    text = soup.get_text(separator=" ")

    # 5. Clean extra spaces
    clean_text = " ".join(text.split())

    return clean_text
