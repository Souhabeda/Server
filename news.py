import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from db import news_collection

# Téléchargement des ressources nécessaires
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Configuration Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Fonction de résumé basé sur mots-clés
def extract_event_summary(text):
    keywords = ["Gold", "Silver", "Bitcoin", "buy", "sell", "increase", "decrease", "market", "crypto", "price", "forex", "bearish", "bullish"]
    sentences = text.split(".")
    important = [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]
    if not important:
        important = sentences[:3]
    return ". ".join(important[:2]) + "."


# Fonction principale pour récupérer les news
def get_news():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.kitco.com/news/digest#crypto")
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    news_items = soup.select("div.DigestNews_newItem__K4a83")

    new_articles = []
    duplicate_count = 0  # compteur pour les doublons

    for item in news_items:
        title_tag = item.find("a")
        title = title_tag.text.strip() if title_tag else "N/A"
        link = f"https://www.kitco.com{title_tag['href']}" if title_tag else "N/A"

        author_tag = item.select_one("span.line-clamp-1.grow-0.px-2.DigestNews_itemSource__MwS0Z")
        author = author_tag.text.strip() if author_tag else "N/A"

        date_tag = item.select_one("p.text-gray-500")
        if date_tag:
            raw_date = date_tag.text.strip()
            match = re.search(r"[A-Za-z]+\s(\w+)", raw_date)
            date = match.group(1) if match else raw_date
        else:
            date = "N/A"

        score = sia.polarity_scores(title)["compound"]
        if score >= 0.05:
            sentiment = "📈"
        elif score <= -0.05:
            sentiment = "📉"
        else:
            sentiment = "⚖️"

        # Évite les doublons
        if news_collection.find_one({"link": link}):
            duplicate_count += 1
            continue

        try:
            driver.get(link)
            time.sleep(5)
            article_soup = BeautifulSoup(driver.page_source, "html.parser")
            content = (
                article_soup.select_one("div.article-body")
                or article_soup.select_one("div.content")
                or article_soup.select_one("section.article-body")
                or article_soup.select_one("article")
                or article_soup.select_one("div.main-content")
            )
            if content:
                full_text = " ".join(p.text.strip() for p in content.find_all("p"))
                summary = extract_event_summary(full_text)

                # Nettoyage : retirer "(Kitco News) - ", ou variantes similaires
                summary = re.sub(r"^\(Kitco News\)\s*[-–—]?\s*", "", summary).strip()
            else:
                summary = "No article content found."
        except Exception as e:
            summary = f"Error fetching summary: {e}"

        news_doc = {
            "title": title,
            "link": link,
            "author": author,
            "date": date,
            "sentiment": sentiment,
            "score": score,
            "summary": summary
        }

        news_collection.insert_one(news_doc)
        new_articles.append(news_doc)

    driver.quit()

    # 📊 Résumé final
    print(f"\n📊 Résumé :")
    print(f"➕ {len(new_articles)} Latest Kitco news updates are now available..")
    print(f"➖ {duplicate_count} Existing Kitco news articles.\n")

    return new_articles
