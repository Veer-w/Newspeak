import asyncio
from abc import ABC, abstractmethod
from typing import Sequence
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import urlparse
import resend
from jinja2 import Environment
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
        .badge-score { font-weight: 700; }
        .score-high { background-color: #dcfce7; color: #15803d; }
        .score-med { background-color: #dbeafe; color: #1d4ed8; }
        .score-low { background-color: #f3f4f6; color: #4b5563; }
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
        .source-link {
            margin-top: 14px;
            font-size: 13px;
        }
        .source-link a {
            color: #4f46e5;
            text-decoration: none;
            font-weight: 600;
        }
        .source-link a:hover { text-decoration: underline; }
        .source-link .label {
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
            margin-right: 6px;
        }
        .footer {
            text-align: center;
            padding: 30px 20px;
            font-size: 12px;
            color: #64748b;
            line-height: 1.5;
        }
        .footer a { color: #4f46e5; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Newspeak Digest</h1>
            <p>Your Daily Machine Learning &amp; AI Intelligence</p>
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
                <span class="badge badge-score score-high">★ {{ "%.1f"|format(item.score) }} / 10.0</span>
                {% elif item.score >= 6.0 %}
                <span class="badge badge-score score-med">★ {{ "%.1f"|format(item.score) }} / 10.0</span>
                {% else %}
                <span class="badge badge-score score-low">★ {{ "%.1f"|format(item.score) }} / 10.0</span>
                {% endif %}
            </div>

            <div class="summary">{{ item.summary }}</div>

            <div class="reason-box">
                <p><strong>Impact:</strong> {{ item.reason }}</p>
            </div>

            <div class="source-link">
                <span class="label">Source</span>
                <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">{{ item.source }} — read the original at {{ item.url | domain }} &rarr;</a>
            </div>
        </div>
        {% endfor %}

        <div class="footer">
            <p>This email was automatically generated and curated by <strong>Newspeak</strong> using Google Gemini.</p>
            <p>&copy; 2026 Newspeak. Powered by Functional Python &amp; uv.</p>
        </div>
    </div>
</body>
</html>
"""

def _domain(url: str) -> str:
    """Pure helper: extract a clean display domain from a URL for the source link."""
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc or url
    except Exception:
        return url


# Jinja2 environment with autoescape enabled to prevent XSS from LLM-generated content
_JINJA_ENV = Environment(autoescape=True)
_JINJA_ENV.filters["domain"] = _domain
_NEWSLETTER_TEMPLATE = _JINJA_ENV.from_string(HTML_TEMPLATE)


def render_newsletter_html(date_str: str, items: Sequence[NewsItem]) -> str:
    """Renders the Jinja2 HTML email template with autoescape. (Pure function)"""
    return _NEWSLETTER_TEMPLATE.render(date=date_str, items=items)


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

    async def send_newsletter(self, date_str: str, items: Sequence[NewsItem], recipients: Sequence[str]) -> bool:
        if not recipients:
            logger.warning("No recipients specified for Resend delivery.")
            return False

        html_content = render_newsletter_html(date_str, items)
        subject = f"Newspeak Daily: Top AI/ML News ({date_str})"

        # Capture the api_key locally to avoid closure over mutable self in executor
        api_key = self.api_key
        sender = self.sender

        try:
            logger.info(f"Sending email via Resend to {len(recipients)} recipients...")
            loop = asyncio.get_running_loop()

            def send_call():
                resend.api_key = api_key
                return resend.Emails.send({
                    "from": sender,
                    "to": list(recipients),
                    "subject": subject,
                    "html": html_content
                })

            result = await loop.run_in_executor(None, send_call)

            # Resend SDK returns an Email object with an `id` on success
            email_id = getattr(result, "id", None)
            if not email_id:
                logger.error(f"Resend API returned unexpected response (no id): {result}")
                return False

            logger.info(f"Email delivered successfully via Resend. Email ID: {email_id}")
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
            loop = asyncio.get_running_loop()

            server = self.server
            port = self.port
            username = self.username
            password = self.password
            from_addr = self.from_addr

            def send_smtp():
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = from_addr
                msg["To"] = ", ".join(recipients)
                msg.attach(MIMEText(html_content, "html", "utf-8"))

                # Port 465 speaks implicit TLS (SMTPS) from the first byte, so it needs
                # SMTP_SSL and must NOT call starttls(). 587/25 connect in plaintext and
                # then upgrade via STARTTLS.
                smtp_factory = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
                with smtp_factory(server, port) as smtp:
                    if port != 465:
                        smtp.starttls()
                    if username and password:
                        smtp.login(username, password)
                    smtp.sendmail(from_addr, list(recipients), msg.as_string())

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
            # Always specify utf-8 to handle Unicode article titles correctly
            self.output_path.write_text(html_content, encoding="utf-8")
            logger.info("Mock newsletter file written successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to write mock newsletter: {e}")
            return False
