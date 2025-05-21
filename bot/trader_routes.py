# routes.py
from flask import Blueprint, jsonify, request
from bot.trader import get_recommendation, place_trade,  get_news_newsapi
from bot.utils import detect_symbol, detect_horizon, detect_language, is_trading_question, META_PROMPT, SYMBOL_TO_NEWS_QUERY, is_greeting, is_news_request, is_advice_request, is_valid_response
from bot.google_chat import ask_gemini 
from dotenv import load_dotenv
import os

load_dotenv()
# === NewsApi ===
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Création d'un Blueprint pour les routes
trader_bp = Blueprint('trading', __name__)

# Route pour obtenir la recommandation de trading
@trader_bp.route('/recommendation/<symbol>', methods=['GET'])
def recommendation(symbol):
    try:
        rec = get_recommendation(symbol)
        data = rec["data"]  # C'est ici qu'on trouve "symbol", "rsi", etc.

        message = (
            f"For the symbol {data['symbol']}, the current price is {data['current_price']:.5f} and "
            f"the predicted price for the next period is {data['predicted_price']:.5f}, indicating  "
            f"a trend {data['trend']}. The RSI indicator is at {data['rsi']:.2f} and the MACD is at {data['macd']:.5f} "
            f"with a signal at {data['macd_signal']:.5f}, showing a MACD trend {data['macd_trend']}. "
            f"The final recommendation is to {data['recommendation']}."
        )

        rec["message"] = message
        return jsonify(rec)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@trader_bp.route('/trade', methods=['POST'])
def trade():
    data = request.json
    message = data.get("message", "").lower()
    language = detect_language(message)

    # Détection automatique de symbol
    symbol = detect_symbol(message)
    
    # Détection de l'action
    if any(word in message for word in ["buy", "acheter", "long"]):
        action = "BUY"
    elif any(word in message for word in ["sell", "vendre", "short"]):
        action = "SELL"
    else:
        error_msg = "Action not recognized." if language == "en" else "Action non reconnue."
        return jsonify({"success": False, "error": error_msg}), 400

    if not symbol:
        error_msg = "Symbol not recognized." if language == "en" else "Symbole non reconnu."
        return jsonify({"success": False, "error": error_msg}), 400

    try:
        result = place_trade(symbol, action)
        success_msg = "Trade placed successfully." if language == "en" else "Ordre passé avec succès."
        return jsonify({"success": True, "message": success_msg, "data": result})
    except Exception as e:
        error_msg = str(e)
        return jsonify({"success": False, "error": error_msg}), 500

@trader_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    
    language = detect_language(message)
    symbol = detect_symbol(message)
    horizon = detect_horizon(message)

    # Répondre aux salutations simples
    if is_greeting(message):
        return jsonify({
            "success": True,
            "language": language,
            "response": (
                "Bonjour 👋 Je suis votre Xpero assistant de trading. Posez-moi une question liée aux marchés !"
                if language == "fr"
                else "Hello 👋 I'm your Xpero trading assistant. Ask me something market-related!"
            )
        })

    try:
        # Priorité aux questions news avec symbole : utiliser NewsAPI en temps réel
        if symbol and is_news_request(message):
            news_query = SYMBOL_TO_NEWS_QUERY.get(symbol, symbol)
            news_articles = get_news_newsapi(news_query, NEWSAPI_KEY, language)
            if not news_articles or (len(news_articles) == 1 and news_articles[0].startswith("Error")):
                msg = "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
                return jsonify({"success": True, "response": msg})
            response_text = (
                f"Voici les dernières nouvelles sur {symbol} :\n" + "\n".join(news_articles)
                if language == "fr"
                else f"Here are the latest news about {symbol}:\n" + "\n".join(news_articles)
            )
            return jsonify({
                "success": True,
                "language": language,
                "response": response_text
            })

        # Questions conseils avec symbole restent Gemini
        if symbol and is_advice_request(message):
            gemini_response = ask_gemini(message, system_prompt=META_PROMPT)
            if not is_valid_response(gemini_response):
                msg = "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
                return jsonify({"success": True, "response": msg})
            return jsonify({
                "success": True,
                "language": language,
                "response": gemini_response
            })

        # Prédiction si symbole détecté
        if symbol:
            rec = get_recommendation(symbol, horizon_days=horizon)
            if language == "fr":
                traduction_recommendation = {"BUY": "Acheter", "SELL": "Vendre", "HOLD": "Conserver"}
                traduction_tendance = {"UP": "à la hausse", "DOWN": "à la baisse", "flat": "stable"}

                recommandation_fr = traduction_recommendation.get(rec['data']['recommendation'], rec['data']['recommendation'])
                tendance_fr = traduction_tendance.get(rec['data']['trend'], rec['data']['trend'])

                response = (
                    f"Prédiction pour {symbol} dans {horizon} jour(s) :\n"
                    f"Prix actuel : {rec['data']['current_price']:.5f}, "
                    f"prix prédit : {rec['data']['predicted_price']:.5f}, "
                    f"tendance : {tendance_fr}.\n"
                    f"Recommandation : {recommandation_fr}."
                )
            else:
                response = (
                    f"Prediction for {symbol} in {horizon} day(s):\n"
                    f"Current price: {rec['data']['current_price']:.5f}, "
                    f"predicted price: {rec['data']['predicted_price']:.5f}, "
                    f"trend: {rec['data']['trend']}.\n"
                    f"Recommendation: {rec['data']['recommendation']}."
                )

            if not is_valid_response(response, symbol=symbol):
                msg = "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
                return jsonify({"success": True, "response": msg})

            return jsonify({
                "success": True,
                "language": language,
                "symbol": symbol,
                "horizon_days": horizon,
                "response": response
            })

        # Questions générales trading
        if is_trading_question(message):
            gemini_response = ask_gemini(message, system_prompt=META_PROMPT)
            if not is_valid_response(gemini_response):
                msg = "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
                return jsonify({"success": True, "response": msg})
            return jsonify({
                "success": True,
                "language": language,
                "response": gemini_response
            })

        # Demande d’actualité générale : utiliser NewsAPI en temps réel
        if is_news_request(message):
            news_articles = get_news_newsapi("crypto OR market", NEWSAPI_KEY, language)
            if not news_articles or (len(news_articles) == 1 and news_articles[0].startswith("Error")):
                msg = "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
                return jsonify({"success": True, "response": msg})
            response_text = (
                "Voici les dernières nouvelles sur la crypto et les marchés aujourd'hui :\n" + "\n".join(news_articles)
                if language == "fr"
                else "Here are the latest crypto and market news today:\n" + "\n".join(news_articles)
            )
            return jsonify({
                "success": True,
                "language": language,
                "response": response_text
            })

        # Demande de conseils générale
        if is_advice_request(message):
            simple_advice_prompt = (
                "You are a friendly trading assistant. Answer simply and clearly with easy-to-understand trading advice, no jargon, in a short paragraph."
                if language == "en"
                else "Tu es un assistant trading sympathique. Réponds simplement, clairement, avec des conseils de trading faciles à comprendre, sans jargon, en un court paragraphe."
            )
            full_prompt = f"{simple_advice_prompt}\nQuestion: {message}\nAnswer:"
            gemini_response = ask_gemini(full_prompt, system_prompt=META_PROMPT)
            if not is_valid_response(gemini_response):
                msg = "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
                return jsonify({"success": True, "response": msg})

            return jsonify({
                "success": True,
                "language": language,
                "response": gemini_response
            })

        # Si rien ne matche
        return jsonify({
            "success": True,
            "response": "Désolé, je suis un assistant de trading." if language == "fr" else "Sorry, I'm a trading assistant."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
