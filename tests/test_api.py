from fastapi.testclient import TestClient
from newspeak.api import app

client = TestClient(app)

def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "newspeak"


def test_preview_mocked() -> None:
    # Trigger preview with mock LLM so it doesn't call the actual Gemini API
    response = client.get("/preview?mock_llm=true")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Newspeak Digest" in html
    assert "[MOCK SUMMARY]" in html
