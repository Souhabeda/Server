# db.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["auth"]

# Collections
users_collection = db["users"]
news_collection = db["news"]
forex_news_collection = db["forex_news"]  

# Test de connexion
try:
    client.admin.command('ping')
    print("✅ Connexion réussie à MongoDB Atlas")
except Exception as e:
    print("❌ Erreur de connexion :", e)
