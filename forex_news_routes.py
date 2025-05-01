# forex_news_routes.py
from flask import Blueprint, jsonify
from db import forex_news_collection
from forex_news import job 

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
            "message": f"{added} nouvelles actualités Forex ajoutées.",
            "added": added
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500