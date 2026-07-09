from abc import ABC, abstractmethod
from typing import Sequence
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import resend
from jinja2 import Template
from newspeak.types import NewsItem

logger = logging.getLogger(__name__)

# Premium HTML email template with custom CSS and clean layout
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Newspeak Daily AI/ML Digest</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
        }
        .container {
            max-width: 680px;
            margin: 0 auto;
            padding: 20px 10px;
        }
        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border-radius: 16px;
            padding: 40px 30px;
            color: white;
            text-align: center;
            margin-bottom: 24px;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.15);
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.025em;
        }
        .header p {
            margin: 8px 0 0 0;
            font-size: 16px;
            color: #e0e7ff;
            font-weight: 500;
        }
        .date {
            display: inline-block;
            background-color: rgba(255, 255, 255, 0.15);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 12px;
            letter-spacing: 0.05em;
        }
        .card {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        .title {
            font-size: 18px;
            font-weight: 700;
            line-height: 1.4;
            margin: 0;
            color: #0f172a;
        }
        .title a {
            color: #0f172a;
            text-decoration: none;
        }
        .title a:hover {
            color: #4f46e5;
            text-decoration: underline;
        }
        .metadata {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin-top: 8px;
        }
        .badge {
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }
        .badge-source {
            background-color: #f1f5f9;
            color: #475569;
        }
        .badge-score {
            font-weight: 700;
        }
        .score-high {
            background-color: #dcfce7;
            color: #15803d;
        }
        .score-med {
            background-color: #dbeafe;
            color: #1d4ed8;
        }
        .score-low {
            background-color: #f3f4f6;
            color: #4b5563;
        }
        .summary {
            font-size: 15px;
            line-height: 1.6;
            color: #334155;
            margin: 12px 0;
        }
        .reason-box {
            background-color: #faf5ff;
            border-left: 3px solid #a855f7;
            padding: 10px 14px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .reason-box p {
            margin: 0;
            font-size: 13px;
            color: #6b21a8;
            font-style: italic;
        }
        .footer {
            text-align: center;
            padding: 30px 20px;
            font-size: 12px;
            color: #64748b;
            line-height: 1.5;
        }
        .footer a {
            color: #4f46e5;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Newspeak Digest</h1>
            <p>Your Daily Machine Learning & AI Intell</p>
            <div class="date">{{ date }}</div>
        </div>

        {% for item in items %}
        <div class="card">
            <div class="title">
                <a href="{{ item.url }}" target="_blank">{{ item.title }}</a>
            </div>
            
            <div class="metadata">
                <span class="badge badge-source">{{ item.source }}</span>
                {% if item.score >= 8.5 %}
                <span class="badge badge-score score-high">★ {{ item.score }} / 10.0</span>
                {% elif item.score >= 6.0 %}
                <span class="badge badge-score score-med">★ {{ item.score }} / 10.0</span>
                {% else %}
                <span class="badge badge-score score-low">★ {{ item.score }} / 10.0</span>
                {% endif %}
            </div>

            <div class="summary">
                {{ item.summary }}
            </div>

            <div class="reason-box">
                <p><strong>Impact:</strong> {{ item.reason }}</p>
            </div>
        </div>
        {% endfor %}

        <div class="footer">
            <p>This email was automatically generated and curated by <strong>Newspeak</strong> using Google Gemini.</p>
            <p>&copy; 2026 Newspeak. Powered by Functional Python & uv.</p>
        </div>
    </div>
</body>
</html>
"""

def render_newsletter_html(date_str: str, items: Sequence[NewsItem]) -> str:
    """Renders the Jinja2 HTML email template. (Pure function)"""
    template = Template(HTML_TEMPLATE)
    return template.render(date=date_str, items=items)


class EmailDelivery(ABC):
    """Abstract Base Class for email delivery."""

    @abstractmethod
    async def send_newsletter(self, date_str: str, items: Sequence[NewsItem], recipients: Sequence[str]) -> bool:
        """Sends the newsletter to the given recipients."""
        pass


class ResendDelivery(EmailDelivery):
    """Delivers email using Resend API."""

    def __init__(self, api_key: str, sender: str = "newspeak@resend.dev"):
        self.api_key = api_key
        self.sender = sender
        resend.api_key = api_key

    async def send_newsletter(self, date_str: str, items: Sequence[NewsItem], recipients: Sequence[str]) -> bool:
        if not recipients:
            logger.warning("No recipients specified for Resend delivery.")
            return False

        html_content = render_newsletter_html(date_str, items)
        subject = f"Newspeak Daily: Top AI/ML News ({date_str})"
        
        try:
            logger.info(f"Sending email via Resend to {len(recipients)} recipients...")
            # Run resend email dispatch in executor to keep it async friendly
            import asyncio
            loop = asyncio.get_event_loop()
            
            def send_call():
                # Resend allows a list of string emails for 'to' parameter
                return resend.Emails.send({
                    "from": self.sender,
                    "to": list(recipients),
                    "subject": subject,
                    "html": html_content
                })
            
            await loop.run_in_executor(None, send_call)
            logger.info("Email delivered successfully via Resend API.")
            return True
        except Exception as e:
            logger.error(f"Resend delivery failed: {e}")
            return False


class SMTPDelivery(EmailDelivery):
    """Delivers email using standard SMTP (e.g. Gmail SMTP, SendGrid SMTP, local mail)."""

    def __init__(self, server: str, port: int, username: str, password: str, from_addr: str):
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr

    async def send_newsletter(self, date_str: str, items: Sequence[NewsItem], recipients: Sequence[str]) -> bool:
        if not recipients:
            logger.warning("No recipients specified for SMTP delivery.")
            return False

        html_content = render_newsletter_html(date_str, items)
        subject = f"Newspeak Daily: Top AI/ML News ({date_str})"

        try:
            logger.info(f"Connecting to SMTP server {self.server}:{self.port}...")
            
            import asyncio
            loop = asyncio.get_event_loop()

            def send_smtp():
                # Build mime message
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = self.from_addr
                msg["To"] = ", ".join(recipients)

                # Attach html part
                msg.attach(MIMEText(html_content, "html"))

                # SMTP login and send
                with smtplib.SMTP(self.server, self.port) as smtp:
                    smtp.starttls()  # Upgrade connection to TLS
                    if self.username and self.password:
                        smtp.login(self.username, self.password)
                    smtp.sendmail(self.from_addr, list(recipients), msg.as_string())

            await loop.run_in_executor(None, send_smtp)
            logger.info("Email delivered successfully via SMTP.")
            return True
        except Exception as e:
            logger.error(f"SMTP delivery failed: {e}")
            return False


class MockDelivery(EmailDelivery):
    """Writes the email to a local HTML file for dry-run/preview testing."""

    def __init__(self, output_path: str = "last_newsletter.html"):
        self.output_path = Path(output_path)

    async def send_newsletter(self, date_str: str, items: Sequence[NewsItem], recipients: Sequence[str]) -> bool:
        logger.info(f"Mock Delivery: Outputting email for {len(recipients)} recipients to {self.output_path.absolute()}")
        try:
            html_content = render_newsletter_html(date_str, items)
            self.output_path.write_text(html_content)
            logger.info("Mock newsletter file written successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to write mock newsletter: {e}")
            return False
