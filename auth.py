from flask import Blueprint, request, jsonify, send_from_directory
import bcrypt, jwt, datetime, os
from db import users_collection
from utils import is_strong_password, generate_strong_password, send_email
from dotenv import load_dotenv
import re
import pycountry
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import cloudinary
import cloudinary.uploader
import hashlib
import time

load_dotenv()


auth_bp = Blueprint('auth', __name__)
SECRET_KEY = os.getenv("JWT_SECRET_KEY")

UPLOAD_FOLDER = "static/profile_pics"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
auth_bp.config = {}  # pour éviter une erreur si config non définie
auth_bp.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

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
        return jsonify({"msg": "All fields are required."}), 400

    # Vérification du format d'email
    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"msg": "Invalid email format."}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "Email already in use."}), 409

    if password != re_password:
        return jsonify({"msg": "Passwords do not match."}), 400

    if not is_strong_password(password):
        return jsonify({
            "msg": "Weak password: minimum 8 characters, including uppercase, lowercase, number, and special character."
        }), 400

    if not re.fullmatch(r"^\+\d{8,15}$", phone):
        return jsonify({
            "msg": "Invalid international phone number (e.g., +21624567890)."
        }), 400

    if not pycountry.countries.get(alpha_2=country.upper()):
        return jsonify({
            "msg": "Invalid country code (e.g., TN, FR, CA)."
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

    return jsonify({"msg": "Account created successfully."}), 201

# LOGIN avec EMAIL + MOT DE PASSE
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email").lower()
    password = data.get("password")
    remember_me = data.get("remember_me", False)

    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"msg": "Invalid email address."}), 401

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
        return jsonify({"msg": "Token missing."}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expired."}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Invalid token."}), 403

    data = request.get_json()
    old_password = data.get("old_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not all([old_password, new_password, confirm_password]):
        return jsonify({"msg": "All fields are required."}), 400

    if new_password != confirm_password:
        return jsonify({"msg": "New passwords do not match."}), 400

    if not is_strong_password(new_password):
        return jsonify({"msg": "Weak password: minimum 8 characters, including uppercase, lowercase, number, and special character."}), 400

    user = users_collection.find_one({"email": email})

    if not user or not bcrypt.checkpw(old_password.encode('utf-8'), user["password"]):
        return jsonify({"msg": "Incorrect old password."}), 401

    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    users_collection.update_one({"email": email}, {"$set": {"password": hashed_pw}})

    return jsonify({"msg": "Password changed successfully."}), 200

# PROTECTED
@auth_bp.route("/protected", methods=["GET"])
def protected():
    token = request.headers.get("Authorization")

    if not token:
        return jsonify({"msg": "Token missing."}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"msg": f"Bienvenue {payload['email']}"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expired."}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Invalid token."}), 403

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
        return jsonify({"msg": "Email not found"}), 404

    new_password = generate_strong_password()
    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_pw}}
    )

    send_email(
        to_email=email,
        subject="Password Reset",
        body=f"Hello {user.get('first_name', '')} {user.get('last_name', '')},\n\nYour new password is{new_password}\n\nPlease log in and change it."
    )

    return jsonify({"msg": "A new password has been sent to your email."}), 200

# Get user connecté
@auth_bp.route("/me", methods=["GET"])
def get_user():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"msg": "Token missing."}), 401
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expired."}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Invalid token."}), 403

    user = users_collection.find_one({"email": email}, {"_id": 0, "password": 0})

    if not user:
        return jsonify({"msg": "User not found."}), 404

    return jsonify({"user": user}), 200

@auth_bp.route('/static/profile_pics/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Update user profile
@auth_bp.route("/me", methods=["PUT"])
def update_profile():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"msg": "Token missing."}), 401

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"].lower()
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Token expired."}), 403
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Invalid token."}), 403

    update_data = {}
    allowed_fields = ["first_name", "last_name", "country", "phone", "gender"]

    # Get form data or JSON
    if request.content_type.startswith('multipart/form-data'):
        data = request.form.to_dict()
    else:
        data = request.get_json()

    for field in allowed_fields:
        if field in data:
            value = data[field]
            if field == "gender" and value not in ["male", "female"]:
                return jsonify({"msg": "Gender must be 'male' or 'female'."}), 400
            if field == "country" and not pycountry.countries.get(alpha_2=value.upper()):
                return jsonify({"msg": "Invalid country code (e.g., TN, FR, CA)."}), 400
            if field == "phone" and not re.fullmatch(r"^\+\d{8,15}$", value):
                return jsonify({"msg": "Invalid international phone number (e.g., +21624567890)."}), 400
            update_data[field] = value.upper() if field == "country" else value

    # 📸 Upload image vers Cloudinary
    if 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and allowed_file(file.filename):
            email_hash = hashlib.md5(email.encode()).hexdigest()
            timestamp = int(time.time())

            try:
                upload_result = cloudinary.uploader.upload(
                    file,
                    folder="user_profiles",
                    public_id=f"{email_hash}_{timestamp}",
                    overwrite=True,
                    resource_type="image"
                )
                update_data['profile_picture'] = upload_result["secure_url"]
            except Exception as e:
                return jsonify({"msg": f"Image upload failed: {str(e)}"}), 500
        else:
            return jsonify({"msg": "Invalid image format. Allowed: png, jpg, jpeg, webp."}), 400

    if not update_data:
        return jsonify({"msg": "No data to update."}), 400

    users_collection.update_one(
        {"email": email},
        {"$set": update_data}
    )

    return jsonify({
        "msg": "Profile updated successfully.",
        "profile_picture": update_data.get("profile_picture")
    }), 200


@auth_bp.route("/delete-account", methods=["DELETE"])
def delete_user():
    try:
        token = request.headers.get("Authorization")

        if not token:
            return jsonify({"msg": "Token missing."}), 401

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            email = payload["email"].lower()
        except jwt.ExpiredSignatureError:
            return jsonify({"msg": "Token expired."}), 403
        except jwt.InvalidTokenError:
            return jsonify({"msg": "Invalid token."}), 403

        data = request.get_json()
        password = data.get("password")

        if not password:
            return jsonify({"msg": "Password required."}), 400

        user = users_collection.find_one({"email": email})

        if not user:
            return jsonify({"msg": "User not found."}), 404

        # Vérification avec bcrypt (comme dans /login)
        if not bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            return jsonify({"msg": "Incorrect password."}), 400

        result = users_collection.delete_one({"email": email})

        if result.deleted_count == 0:
            return jsonify({"msg": "Error occurred during deletion."}), 500

        return jsonify({"msg": "Account deleted successfully."}), 200

    except Exception as e:
        print(f"[ERROR] {e}")  # log dans la console
        return jsonify({"msg": f"Internal error: {str(e)}"}), 500


