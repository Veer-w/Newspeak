import asyncio
import argparse
import sys
import logging
from newspeak.config import load_config
from newspeak.llm import build_llm_provider
from newspeak.delivery import build_delivery_client
from newspeak.pipeline import run_newsletter_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("newspeak.main")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Newspeak Daily AI/ML Digest Ingest and Delivery CLI")
    parser.add_argument("--dry-run", "-d", action="store_true",
                        help="Run pipeline and write HTML to a local file, skipping email dispatch.")
    parser.add_argument("--mock-llm", "-m", action="store_true",
                        help="Use mock LLM ranking instead of the live Gemini API.")
    parser.add_argument("--output", "-o", type=str, default="last_newsletter.html",
                        help="Filename to write the HTML output to when in dry-run mode.")
    parser.add_argument("--serve", "-s", action="store_true",
                        help="Start the FastAPI HTTP API server.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the API server to.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the API server on.")
    return parser


async def _run_pipeline(args: argparse.Namespace) -> None:
    """Runs the newsletter pipeline CLI mode."""
    config = load_config()

    # Determine LLM Provider (backend selection + fallback wiring lives in the factory).
    try:
        llm_provider = build_llm_provider(config, mock=args.mock_llm)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Determine Email Delivery Client (provider selection lives in the factory).
    try:
        delivery_client = build_delivery_client(config, dry_run=args.dry_run, output_path=args.output)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Validate recipients
    if not args.dry_run and not config.recipients:
        logger.error("No recipients configured in RECIPIENTS env variable. Pipeline aborted.")
        sys.exit(1)

    # Execute pipeline
    try:
        success = await run_newsletter_pipeline(
            config=config,
            llm_provider=llm_provider,
            delivery_client=delivery_client
        )
        if not success:
            logger.error("Pipeline completed with errors.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """
    Synchronous entrypoint. Parses CLI args first, then either:
      - Launches uvicorn (blocking, sync) for --serve mode, OR
      - Runs the async pipeline via asyncio.run()
    Keeping arg parsing and uvicorn.run() outside asyncio.run() avoids
    the 'event loop is already running' error when uvicorn creates its own loop.
    """
    args = _build_arg_parser().parse_args()

    if args.serve:
        import uvicorn
        logger.info(f"Starting API server on {args.host}:{args.port}...")
        uvicorn.run("newspeak.api:app", host=args.host, port=args.port, log_level="info")
    else:
        asyncio.run(_run_pipeline(args))


if __name__ == "__main__":
    main()
