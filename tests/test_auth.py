from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from caption_worker.auth import require_api_key
from caption_worker.config import get_settings


def test_api_key_is_optional(monkeypatch) -> None:
    monkeypatch.setenv("CAPTION_WORKER_API_KEY", "")
    get_settings.cache_clear()

    app = FastAPI()

    @app.get("/", dependencies=[Depends(require_api_key)])
    def route() -> dict[str, bool]:
        return {"ok": True}

    assert TestClient(app).get("/").status_code == 200


def test_api_key_rejects_missing_token(monkeypatch) -> None:
    monkeypatch.setenv("CAPTION_WORKER_API_KEY", "secret")
    get_settings.cache_clear()

    app = FastAPI()

    @app.get("/", dependencies=[Depends(require_api_key)])
    def route() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer secret"}).status_code == 200
