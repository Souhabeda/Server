from flask import Blueprint, request, jsonify
import bcrypt, jwt, datetime, os
from db import users_collection
from utils import is_strong_password, generate_strong_password, send_email
from dotenv import load_dotenv
import re
import pycountry
from werkzeug.security import check_password_hash
from datetime import datetime, timedelta
from bson.objectid import ObjectId

load_dotenv()

auth_bp = Blueprint('auth', __name__)
SECRET_KEY = os.getenv("JWT_SECRET_KEY")

#signup
@auth_bp.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    email = data.get("email").lower()
    password = data.get("password")
    re_password = data.get("re_password")
    country = data.get("country")
    phone = data.get("phone")

    if not all([first_name, last_name, email, password, re_password, country, phone]):
        return jsonify({"msg": "Tous les champs sont obligatoires"}), 400

    # Vérification du format d'email
    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"msg": "Format d'email invalide"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "Email déjà utilisé"}), 409

    if password != re_password:
        return jsonify({"msg": "Les mots de passe ne correspondent pas"}), 400

    if not is_strong_password(password):
        return jsonify({
            "msg": "Mot de passe faible. Minimum 8 caractères, majuscules, minuscules, chiffres et caractère spécial."
        }), 400

    if not re.fullmatch(r"^\+\d{8,15}$", phone):
        return jsonify({
            "msg": "Numéro de téléphone international invalide (ex: +21624567890)"
        }), 400

    if not pycountry.countries.get(alpha_2=country.upper()):
        return jsonify({
            "msg": "Code pays invalide (ex: TN, FR, CA)"
        }), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    users_collection.insert_one({
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": hashed_pw,
        "country": country.upper(),
        "phone": phone
    })

    return jsonify({"msg": "Compte créé avec succès"}), 201

# LOGIN avec EMAIL + MOT DE PASSE
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email").lower()
    password = data.get("password")
    remember_me = data.get("remember_me", False)

    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"msg": "Invalid email address"}), 401

    if not bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return jsonify({"msg": "Incorrect password"}), 401

    # Mettre à jour la date de dernière connexion (UTC)
    now = datetime.utcnow()
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": now}}
    )

    exp_time = now + (timedelta(days=7) if remember_me else timedelta(hours=2))


    token = jwt.encode({
        "email": email,
        "exp": exp_time
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "token": token,
        "remember_me": remember_me,
        "last_login": now.isoformat() + "Z"  # ISO format (facile à manipuler côté frontend)
    }), 200


# CHANGER MOT DE PASSE
@auth_bp.route("/change-password", methods=["POST"])
def change_password():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"msg": "Token manquant"}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expiré"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Token invalide"}), 403

    data = request.get_json()
    old_password = data.get("old_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not all([old_password, new_password, confirm_password]):
        return jsonify({"msg": "Tous les champs sont obligatoires"}), 400

    if new_password != confirm_password:
        return jsonify({"msg": "Les nouveaux mots de passe ne correspondent pas"}), 400

    if not is_strong_password(new_password):
        return jsonify({"msg": "Mot de passe faible. Minimum 8 caractères, majuscules, minuscules, chiffres et caractère spécial."}), 400

    user = users_collection.find_one({"email": email})

    if not user or not bcrypt.checkpw(old_password.encode('utf-8'), user["password"]):
        return jsonify({"msg": "Ancien mot de passe incorrect"}), 401

    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    users_collection.update_one({"email": email}, {"$set": {"password": hashed_pw}})

    return jsonify({"msg": "Mot de passe changé avec succès"}), 200

# PROTECTED
@auth_bp.route("/protected", methods=["GET"])
def protected():
    token = request.headers.get("Authorization")

    if not token:
        return jsonify({"msg": "Token manquant"}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"msg": f"Bienvenue {payload['email']}"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expiré"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Token invalide"}), 403

# GENERATE PASSWORD
@auth_bp.route("/generate-password", methods=["GET"])
def generate_password():
    password = generate_strong_password()
    return jsonify({"password": password}), 200

# FORGOT PASSWORD
@auth_bp.route("/forgetPassword", methods=["POST"])
def forget_password():
    data = request.get_json()
    email = data.get("email")

    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"msg": "Email introuvable"}), 404

    new_password = generate_strong_password()
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_pw}}
    )

    send_email(
        to_email=email,
        subject="Réinitialisation de votre mot de passe",
        body=f"Bonjour {user.get('first_name', '')} {user.get('last_name', '')},\n\nVotre nouveau mot de passe est : {new_password}\n\nMerci de vous reconnecter et de le modifier."
    )

    return jsonify({"msg": "Un nouveau mot de passe a été envoyé par e-mail."}), 200

# Get user connecté
@auth_bp.route("/me", methods=["GET"])
def get_user():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"msg": "Token manquant"}), 401
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expiré"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Token invalide"}), 403

    user = users_collection.find_one({"email": email}, {"_id": 0, "password": 0})

    if not user:
        return jsonify({"msg": "Utilisateur non trouvé"}), 404

    return jsonify({"user": user}), 200

#Update user profile
@auth_bp.route("/me", methods=["PUT"])
def update_profile():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"msg": "Token manquant"}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expiré"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Token invalide"}), 403

    data = request.get_json()
    allowed_fields = ["first_name", "last_name", "country", "phone", "gender"]

    if "gender" in data and data["gender"] not in ["male", "female"]:
        return jsonify({"msg": "Le genre doit être 'male' ou 'female'"}), 400

    if "country" in data and not pycountry.countries.get(alpha_2=data["country"].upper()):
        return jsonify({"msg": "Code pays invalide (ex: TN, FR, CA)"}), 400

    if "phone" in data and not re.fullmatch(r"^\+\d{8,15}$", data["phone"]):
        return jsonify({"msg": "Numéro de téléphone international invalide (ex: +21624567890)"}), 400

    update_data = {key: data[key] for key in allowed_fields if key in data}

    if not update_data:
        return jsonify({"msg": "Aucune donnée à mettre à jour"}), 400

    if "country" in update_data:
        update_data["country"] = update_data["country"].upper()

    users_collection.update_one(
        {"email": email},
        {"$set": update_data}
    )

    return jsonify({"msg": "Profil mis à jour avec succès"}), 200

# Delete Profile avec vérification du mot de passe
@auth_bp.route("/delete-account", methods=["DELETE"])
def delete_user():
    token = request.headers.get("Authorization")

    if not token:
        return jsonify({"msg": "Token manquant"}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expiré"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Token invalide"}), 403

    data = request.get_json()
    password = data.get("password")

    if not password:
        return jsonify({"msg": "Mot de passe requis"}), 400

    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"msg": "Utilisateur non trouvé"}), 404

    if not check_password_hash(user["password"], password):
        return jsonify({"msg": "Mot de passe incorrect"}), 400

    result = users_collection.delete_one({"email": email})

    if result.deleted_count == 0:
        return jsonify({"msg": "Erreur lors de la suppression"}), 500

    return jsonify({"msg": "Compte supprimé avec succès"}), 200
