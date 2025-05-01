from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from datetime import datetime
import time
from db import forex_news_collection


FOREX_FACTORY_URL = "https://www.forexfactory.com/calendar"
TARGET_CURRENCIES = ["USD", "AUD", "EUR"]  # 🔥 Devises ciblées

def scroll_to_bottom(page, pause_time=2):
    """Fait défiler la page jusqu'en bas."""
    last_height = page.evaluate("() => document.body.scrollHeight")
    while True:
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(pause_time)
        new_height = page.evaluate("() => document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def fetch_forex_factory_data():
    """Scrape le calendrier ForexFactory."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        print(f"🌐 Accès à {FOREX_FACTORY_URL}...")
        page.goto(FOREX_FACTORY_URL, timeout=60000)
        page.wait_for_selector(".calendar__table", timeout=20000)

        scroll_to_bottom(page)

        soup = BeautifulSoup(page.content(), 'html.parser')
        calendar_table = soup.find('table', {'class': 'calendar__table'})

        if not calendar_table:
            print("❌ Tableau du calendrier introuvable. La structure HTML a changé.")
            browser.close()
            return []

        rows = calendar_table.find_all('tr', {'class': ['calendar__row', 'calendar_row']})
        data = []
        current_date = None

        header_date = soup.select_one('div.calendar__header-title')
        if header_date:
            try:
                year_in_header = int(header_date.text.strip().split()[-1])
            except ValueError:
                year_in_header = datetime.now().year
        else:
            year_in_header = datetime.now().year

        for row in rows:
            date_elem = row.find('td', {'class': 'calendar__date'})
            if date_elem and date_elem.text.strip():
                try:
                    full_date_str = f"{date_elem.text.strip()} {year_in_header}"
                    current_date = datetime.strptime(full_date_str, "%a %b %d %Y").date()
                except Exception as e:
                    print(f"⚠️ Erreur parsing date : {e}")
                    current_date = None

            if not current_date:
                continue

            time_elem = row.find('td', {'class': 'calendar__time'})
            currency_elem = row.find('td', {'class': 'calendar__currency'})
            impact_elem = row.find('td', {'class': 'calendar__impact'})
            event_elem = row.find('td', {'class': 'calendar__event'})
            actual_elem = row.find('td', {'class': 'calendar__actual'})
            forecast_elem = row.find('td', {'class': 'calendar__forecast'})
            previous_elem = row.find('td', {'class': 'calendar__previous'})

            if not event_elem:
                continue

            data.append({
                "Date": str(current_date),
                "Time": time_elem.text.strip() if time_elem else "N/A",
                "Currency": currency_elem.text.strip() if currency_elem else "N/A",
                "Impact": impact_elem.find('span')['title'] if impact_elem and impact_elem.find('span') else "N/A",
                "Event": event_elem.text.strip(),
                "Actual": actual_elem.text.strip() if actual_elem else "N/A",
                "Forecast": forecast_elem.text.strip() if forecast_elem else "N/A",
                "Previous": previous_elem.text.strip() if previous_elem else "N/A"
            })

        browser.close()
        return data

def format_impact(impact_text):
    """Retourne l'icône, label, et couleur ANSI basée sur l'impact."""
    impact_text = impact_text.lower()

    if "low" in impact_text:
        return ("🟡", "Low Impact", "\033[93m")  # Jaune clair
    elif "medium" in impact_text:
        return ("🟠", "Medium Impact", "\033[38;5;208m")  # Orange vif
    elif "high" in impact_text:
        return ("🔴", "High Impact", "\033[91m")  # Rouge
    elif "non-economic" in impact_text:
        return ("⚪", "Non-Economic", "\033[90m")  # Gris
    else:
        return ("❓", "Unknown Impact", "\033[0m")  # Par défaut

def job():
    """Tâche de scraping et de stockage."""
    print(f"📥 Lancement du scraping à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    news_data = fetch_forex_factory_data()

    if news_data:
        dates = [datetime.strptime(event['Date'], '%Y-%m-%d') for event in news_data]
        date_min = min(dates).strftime('%Y-%m-%d')
        date_max = max(dates).strftime('%Y-%m-%d')

        print(f"🗓 Période du calendrier : {date_min} → {date_max}")
        print(f"✅ {len(news_data)} événements récupérés avant filtrage.\n")

        # 🔥 Filtrer uniquement USD, AUD, EUR
        filtered_news = [news for news in news_data if news['Currency'] in TARGET_CURRENCIES]

        print(f"🎯 {len(filtered_news)} événements après filtrage ({', '.join(TARGET_CURRENCIES)}).\n")

        new_events_count = 0
        existing_events_count = 0

        for news in filtered_news:
            icon, label, color = format_impact(news['Impact'])
            phrase = (f"{icon} [{news['Currency']}] {news['Event']} "
                      f"({news['Date']} {news['Time']}) - {color}{label}\033[0m")
            print(phrase)

            # 🔎 Vérifier si l'événement existe déjà (par Date + Time + Event + Currency)
            existing = forex_news_collection.find_one({
                "Date": news["Date"],
                "Time": news["Time"],
                "Currency": news["Currency"],
                "Event": news["Event"]
            })

            if existing:
                existing_events_count += 1
            else:
                forex_news_collection.insert_one(news)
                new_events_count += 1

        print(f"\n📊 Résumé :")
        print(f"➕ {new_events_count} nouveaux forex news ajoutés.")
        print(f"➖ {existing_events_count} forex news déjà existants.")

    else:
        print("❌ Aucune donnée trouvée.")


def main():
    """Fonction principale."""
    job()  # ➔ Scraper une seule fois et terminer

if __name__ == "__main__":
    main()
