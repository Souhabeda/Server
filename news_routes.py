# kitco news_routes.py
from flask import Blueprint, jsonify
from db import news_collection
from news import get_news
from flask_socketio import SocketIO  # Import socketio ici


def create_news_bp(socketio):

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
        

    # Route pour obtenir les dernières nouvelles et envoyer les mises à jour via WebSocket
    @news_bp.route('/get_latest_news', methods=['GET'])
    def get_latest_news():
        try:
            new_articles = get_news()  # Appelle ta fonction pour récupérer les nouvelles
            if new_articles:
                # Si des nouvelles sont récupérées, envoie-les via WebSocket
                socketio.emit('new_news_update', {'articles': new_articles})
                return jsonify({"message": "News fetched successfully!"}), 200  # Réponse JSON correcte
            else:
                return jsonify({"message": "No new news found."}), 200  # Cas où aucune nouvelle n'est trouvée
        except Exception as e:
            return jsonify({"error": str(e)}), 500  # En cas d'erreur
        
    return news_bp