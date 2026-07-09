import argparse
import asyncio
import sys
import logging
from newspeak.config import load_config
from newspeak.llm import GeminiProvider, MockLLMProvider
from newspeak.delivery import ResendDelivery, SMTPDelivery, MockDelivery
from newspeak.pipeline import run_newsletter_pipeline

# Configure logging style
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("newspeak.main")

async def main() -> None:
    parser = argparse.ArgumentParser(description="Newspeak Daily AI/ML Digest Ingest and Delivery CLI")
    parser.add_argument(
        "--dry-run", "-d", 
        action="store_true", 
        help="Run pipeline and write rendering to a local HTML file, skipping email dispatch."
    )
    parser.add_argument(
        "--mock-llm", "-m", 
        action="store_true", 
        help="Use mock ranking and summarization instead of querying the live Gemini API."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="last_newsletter.html",
        help="Filename to write the HTML output to when running in dry-run mode."
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Start the FastAPI HTTP API server instead of running the pipeline immediately."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The host to bind the API server to."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="The port to run the API server on."
    )
    args = parser.parse_args()

    # If serving the API, launch uvicorn immediately and return
    if args.serve:
        import uvicorn
        logger.info(f"Starting API server on {args.host}:{args.port}...")
        uvicorn.run("newspeak.api:app", host=args.host, port=args.port, log_level="info")
        return

    # Load configuration
    config = load_config()

    # Determine LLM Provider
    if args.mock_llm:
        logger.info("Initializing Mock LLM Provider...")
        llm_provider = MockLLMProvider()
    else:
        if not config.gemini_api_key:
            logger.error("GEMINI_API_KEY environment variable is not set. Use --mock-llm for local dry runs.")
            sys.exit(1)
        logger.info("Initializing Gemini LLM Provider...")
        llm_provider = GeminiProvider(api_key=config.gemini_api_key)

    # Determine Email Delivery Client
    if args.dry_run:
        logger.info(f"Initializing Mock Delivery Client (outputting to {args.output})...")
        delivery_client = MockDelivery(output_path=args.output)
    else:
        # Default to Resend if API key is present
        if config.resend_api_key:
            logger.info("Initializing Resend Delivery Client...")
            delivery_client = ResendDelivery(
                api_key=config.resend_api_key,
                sender=config.smtp_from  # Using configured sender email
            )
        # Fallback to SMTP if SMTP server details are set
        elif config.smtp_server:
            logger.info(f"Initializing SMTP Delivery Client ({config.smtp_server}:{config.smtp_port})...")
            delivery_client = SMTPDelivery(
                server=config.smtp_server,
                port=config.smtp_port,
                username=config.smtp_username,
                password=config.smtp_password,
                from_addr=config.smtp_from
            )
        else:
            logger.error("No valid email configuration found (RESEND_API_KEY or SMTP_SERVER must be set). Running dry-run instead.")
            delivery_client = MockDelivery(output_path=args.output)

    # Validate recipients list if not in dry-run
    if not args.dry_run and not config.recipients:
        logger.error("No recipient emails configured in RECIPIENTS env variable. Pipeline aborted.")
        sys.exit(1)

    # Execute end-to-end pipeline
    try:
        success = await run_newsletter_pipeline(
            config=config,
            llm_provider=llm_provider,
            delivery_client=delivery_client
        )
        if not success:
            logger.error("Pipeline run completed with errors.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
