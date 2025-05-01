import secrets
import string
import re
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


# Charger les variables d'environnement
load_dotenv()

# Vérifie la force d'un mot de passe
def is_strong_password(password):
    if (len(password) < 8 or
        not re.search(r"\d", password) or
        not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
        return False
    return True

# Génère un mot de passe fort
def generate_strong_password(length=12):
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()"
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        if is_strong_password(password):
            return password

def send_email(to_email, subject, body):
    message = Mail(
        from_email=os.getenv("EMAIL_USER"),
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )

    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(f"Email envoyé avec le statut : {response.status_code}")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {e}")