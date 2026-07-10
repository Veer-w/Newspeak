from datetime import datetime, timezone
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from newspeak.config import load_config
from newspeak.llm import build_llm_provider, select_top_candidates
from newspeak.delivery import build_delivery_client, render_newsletter_html
from newspeak.pipeline import ingest_all_sources, deduplicate_articles, run_newsletter_pipeline

logger = logging.getLogger("newspeak.api")

app = FastAPI(
    title="Newspeak API",
    description="HTTP APIs for Newspeak AI/ML Newsletter curation and delivery",
    version="0.1.0"
)

@app.get("/health")
async def health_check():
    """Simple healthcheck endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "newspeak"
    }


@app.get("/preview", response_class=HTMLResponse)
async def preview_newsletter(
    mock_llm: bool = Query(
        default=False, 
        description="If true, uses mock summaries instead of live Gemini API calls."
    )
):
    """
    Ingests feed sources, deduplicates them, curates the top 10 articles (live or mocked),
    and returns the rendered HTML newsletter directly.
    """
    config = load_config()
    
    # 1. Ingestion
    raw_articles = await ingest_all_sources(config)
    if not raw_articles:
        raise HTTPException(status_code=500, detail="Failed to ingest articles from feeds.")
        
    # 2. Deduplication
    curated_candidates = deduplicate_articles(raw_articles)

    # 3. Heuristic pre-ranking + LLM curator selection
    try:
        provider = build_llm_provider(config, mock=mock_llm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    candidates = select_top_candidates(curated_candidates, config.keywords, config.llm_max_candidates)
    top_news = await provider.rank_and_summarize(candidates)
    if not top_news:
        raise HTTPException(status_code=500, detail="LLM curation returned no items.")
        
    # 4. Render HTML response
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    html_content = render_newsletter_html(date_str, top_news)
    return HTMLResponse(content=html_content)


@app.post("/trigger")
async def trigger_pipeline(
    mock_llm: bool = Query(default=False, description="If true, uses mock LLM provider."),
    dry_run: bool = Query(default=False, description="If true, mocks email delivery.")
):
    """
    Triggers the end-to-end news newsletter pipeline:
    Ingests articles -> Deduplicates -> Summarizes -> Dispatches emails.
    """
    config = load_config()

    # Configure LLM provider (backend selection + fallback wiring lives in the factory)
    try:
        provider = build_llm_provider(config, mock=mock_llm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Configure Delivery client (provider selection lives in the factory)
    try:
        delivery_client = build_delivery_client(config, dry_run=dry_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not dry_run and not config.recipients:
        raise HTTPException(status_code=400, detail="No email recipients configured.")

    # Execute pipeline
    try:
        success = await run_newsletter_pipeline(
            config=config,
            llm_provider=provider,
            delivery_client=delivery_client
        )
        if not success:
            raise HTTPException(status_code=500, detail="Pipeline completed with errors.")
        return {"status": "success", "message": "Newsletter pipeline ran and emails delivered successfully."}
    except Exception as e:
        logger.error(f"Error during triggered pipeline run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
