# forex_news_routes.py
from flask import Blueprint, jsonify
from db import forex_news_collection
from forex_news import job 
from datetime import datetime, timedelta
from pymongo import DESCENDING


forex_news_bp = Blueprint('forex_news', __name__)

@forex_news_bp.route("/forex-news", methods=["GET"])
def get_forex_news():
    try:
        # Récupérer toutes les news triées par Date décroissante
        news_list = list(forex_news_collection.find().sort("Date", -1))

        # Nettoyer l'ObjectId de MongoDB
        for news in news_list:
            news["_id"] = str(news["_id"])

        return jsonify({"forex_news": news_list}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@forex_news_bp.route("/forex-news/update", methods=["POST"])
def update_forex_news():
    try:
        old_count = forex_news_collection.count_documents({})
        job()  # Lance ton scraping + insertion en base
        new_count = forex_news_collection.count_documents({})
        added = new_count - old_count

        return jsonify({
            "message": f"{added} New Forex news added.",
            "added": added
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# 📣 Route pour récupérer les dernières news ajoutées
@forex_news_bp.route("/forex-news/new", methods=["GET"])
def get_new_forex_news():
    try:
        # Récupérer les actualités créées dans les dernières 24 heures
        now = datetime.now()
        time_threshold = now - timedelta(days=1)  # 24 heures en arrière

        # Convertir `time_threshold` en ISO format pour la comparaison
        new_news = list(forex_news_collection.find({
            "created_at": {"$gt": time_threshold}
        }).sort("created_at", DESCENDING).limit(5))

        for news in new_news:
            news["_id"] = str(news["_id"])

        return jsonify({"new_forex_news": new_news}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500