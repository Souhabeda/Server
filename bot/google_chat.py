import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

def ask_gemini(message: str, system_prompt: str = "") -> str:
    try:
        # Préparer le contenu de la requête : system_prompt en début si fourni
        contents = []
        if system_prompt:
            contents.append({"text": system_prompt})
        contents.append({"text": message})

        response = client.models.generate_content(
            model="gemini-2.0-flash",  # ou un autre modèle valide que tu auras listé
            contents=contents
        )
        return response.text.strip()

    except Exception as e:
        print(f"[Gemini Error] {type(e).__name__}: {e}")
        return "Erreur avec l'API Gemini. Réessaie plus tard."
