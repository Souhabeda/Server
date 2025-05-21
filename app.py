# app.py
import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_socketio import SocketIO
from auth import auth_bp
from contact import contact_bp
from forex_news_routes import forex_news_bp
from news import get_news
from news_routes import create_news_bp
from forex_news import job
from trading_mt5_routes import routes as trading_mt5_routes
from bot.trader_routes import trader_bp 
from threading import Thread 
from flask import Flask
from flask_socketio import SocketIO
from binance_stream import init_socketio, start_binance_stream  
from prediction_routes import prediction_route
from binance_stream import init_socketio, start_binance_stream

load_dotenv()

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins=[
     "http://localhost:3000",
    "https://trading-platform-inky.vercel.app",
    "https://9b1f-197-31-137-61.ngrok-free.app"
])

CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "https://trading-platform-inky.vercel.app",
    "https://a784-197-27-118-162.ngrok-free.app" , #Forwarding  ngrok  -> http://localhost:5000
]}}, supports_credentials=True)

# Créer le Blueprint news sans provoquer d'importation circulaire
news_bp = create_news_bp(socketio)
app.register_blueprint(auth_bp)
app.register_blueprint(contact_bp)
app.register_blueprint(news_bp)
app.register_blueprint(forex_news_bp)  
app.register_blueprint(trading_mt5_routes)
app.register_blueprint(trader_bp)
app.register_blueprint(prediction_route)

if __name__ == "__main__":
    print("🔁 Automatically scraping the latest Kitco news on startup...")
    get_news()
    print("\n🔁 Automatically scraping Forex Factory events on startup...")
    job()

    print("\n🔌 Initializing SocketIO for Binance stream...")
    init_socketio(socketio)

    # 🔁 Lancement du stream Binance dans un thread séparé
    thread = Thread(target=start_binance_stream)
    thread.daemon = True
    thread.start()

    # ✅ On utilise uniquement socketio.run()
    socketio.run(app, host="0.0.0.0", port=5000)


    