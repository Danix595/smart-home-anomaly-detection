from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.main import app
from src.core import config
from src.db.database import init_db
from src.models import detector
from src.models.train_isolation_forest import train_and_save
from src.simulate_yearly import simulate_yearly_data


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Entrena un Isolation Forest ligero una sola vez por sesión de tests."""
    model_dir = tmp_path_factory.mktemp("models")
    df = simulate_yearly_data(days=14, seed=42, end_date="2026-07-02")
    train_and_save(df, contamination=0.004, n_estimators=100, max_samples=1024, random_state=42, model_dir=model_dir)
    return model_dir


@pytest.fixture()
def isolated_env(tmp_path: Path, trained_model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Aísla BBDD SQLite, secretos, cooldown y caché del modelo para cada test."""
    db_path = tmp_path / "test_domotica.db"
    monkeypatch.setattr(config, "DB_FILE", str(db_path))
    monkeypatch.setattr(config, "SECRET_API_KEY", "test-secret-api-key")
    monkeypatch.setattr(config, "COOLDOWN", 300)
    monkeypatch.setattr(config, "BOT_TOKEN", None)
    monkeypatch.setattr(config, "CHAT_ID", None)
    monkeypatch.setattr(detector, "MODEL_DIR", trained_model_dir)
    detector.reset_model_cache()
    init_db()
    yield
    detector.reset_model_cache()


@pytest.fixture()
def api_client(isolated_env):
    with TestClient(app) as client:
        yield client
