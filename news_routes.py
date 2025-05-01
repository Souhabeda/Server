# news_routes.py
from flask import Blueprint, jsonify
from db import news_collection
from news import get_news

news_bp = Blueprint("news", __name__)

@news_bp.route("/news", methods=["GET"])
def fetch_news():
    try:
        news = list(news_collection.find({}, {"_id": 0}))
        return jsonify(news), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@news_bp.route("/news/update", methods=["POST"])
def update_news():
    try:
        new = get_news()  
        return jsonify({"new_articles": new, "added": len(new)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
