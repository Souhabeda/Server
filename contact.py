# contact.py
from flask import Blueprint, request, jsonify
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv()

contact_bp = Blueprint('contact', __name__)


@contact_bp.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')

    if not all([name, email, subject, message]):
        return jsonify({"error": "All fields are required"}), 400

    # Contenu de l’email
    email_message = Mail(
        from_email=os.getenv('EMAIL_USER'),
        to_emails=os.getenv('EMAIL_USER'),  
        subject=f"Contact Form: {subject}",
        html_content=f"""
            <p><strong>From:</strong> {name} ({email})</p>
            <p><strong>Subject:</strong> {subject}</p>
            <p><strong>Message:</strong><br>{message}</p>
        """
    )

    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        response = sg.send(email_message)
        print("=== SENDGRID RESPONSE ===")
        print(response.status_code)
        print("Body:", response.body)
        print("Headers:", response.headers)
        print("=========================")
        return jsonify({"message": "Message sent via email"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to send email"}), 500

