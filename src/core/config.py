import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None

# Sin valor por defecto inseguro: si no se define API_KEY, los endpoints protegidos rechazarán el acceso.
SECRET_API_KEY = os.getenv("API_KEY", "").strip()

DB_FILE = os.getenv("DB_FILE", str(BASE_DIR / "domotica.db"))

try:
    _cooldown = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "300"))
    COOLDOWN = _cooldown if _cooldown >= 0 else 300
except ValueError:
    COOLDOWN = 300
