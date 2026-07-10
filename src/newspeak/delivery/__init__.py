import logging
from newspeak.config import Config
from newspeak.delivery.email import (
    EmailDelivery,
    ResendDelivery,
    SMTPDelivery,
    MockDelivery,
    render_newsletter_html,
)

logger = logging.getLogger(__name__)


def _make_resend(config: Config) -> ResendDelivery:
    return ResendDelivery(api_key=config.resend_api_key, sender=config.smtp_from)


def _make_smtp(config: Config) -> SMTPDelivery:
    return SMTPDelivery(
        server=config.smtp_server,
        port=config.smtp_port,
        username=config.smtp_username,
        password=config.smtp_password,
        from_addr=config.smtp_from,
    )


def build_delivery_client(
    config: Config,
    dry_run: bool = False,
    output_path: str = "last_newsletter.html",
) -> EmailDelivery:
    """Select the email delivery backend from config (single source of truth).

    `EMAIL_PROVIDER` controls the choice:
      - "smtp":   force standard SMTP (e.g. Gmail) — errors if SMTP_SERVER is unset.
      - "resend": force the Resend API — errors if RESEND_API_KEY is unset.
      - "auto" (default): use Resend if RESEND_API_KEY is set, else SMTP, else mock.
    `dry_run=True` always uses MockDelivery (writes HTML to a local file).
    """
    if dry_run:
        return MockDelivery(output_path=output_path)

    provider = (config.email_provider or "auto").strip().lower()

    if provider == "resend":
        if not config.resend_api_key:
            raise ValueError("EMAIL_PROVIDER=resend but RESEND_API_KEY is not set.")
        logger.info("Email provider: Resend (explicit).")
        return _make_resend(config)

    if provider == "smtp":
        if not config.smtp_server:
            raise ValueError("EMAIL_PROVIDER=smtp but SMTP_SERVER is not set.")
        logger.info(f"Email provider: SMTP (explicit) — {config.smtp_server}:{config.smtp_port}.")
        return _make_smtp(config)

    # auto
    if config.resend_api_key:
        logger.info("Email provider: Resend (auto — RESEND_API_KEY set). Set EMAIL_PROVIDER=smtp to force SMTP.")
        return _make_resend(config)
    if config.smtp_server:
        logger.info(f"Email provider: SMTP (auto) — {config.smtp_server}:{config.smtp_port}.")
        return _make_smtp(config)

    logger.warning("No email delivery method configured — writing to a local file instead.")
    return MockDelivery(output_path=output_path)


__all__ = [
    "EmailDelivery",
    "ResendDelivery",
    "SMTPDelivery",
    "MockDelivery",
    "render_newsletter_html",
    "build_delivery_client",
]
