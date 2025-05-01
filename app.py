# app.py
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from auth import auth_bp
from contact import contact_bp
from news_routes import news_bp
from forex_news_routes import forex_news_bp  
from news import get_news
from forex_news import job 
from trading_mt5_routes import routes as trading_mt5_routes


load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://ton-site.vercel.app"]}})


# Enregistrement des Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(contact_bp)
app.register_blueprint(news_bp)
app.register_blueprint(forex_news_bp)  
app.register_blueprint(trading_mt5_routes)


if __name__ == "__main__":
    print("🔁 Scraping automatique des dernières news Kitco au démarrage...")
    get_news()  # Scraping Kitco
    print("\n🔁 Scraping automatique des événements Forex Factory au démarrage...")
    job()  # Scraping Forex
    app.run(debug=True, port=5000)
