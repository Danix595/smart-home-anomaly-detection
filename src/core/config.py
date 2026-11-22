import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SECRET_API_KEY = os.getenv("API_KEY", "CLAVE_NO_CONFIGURADA")

DB_FILE = "domotica.db"

try:
    COOLDOWN = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "300"))
except ValueError:
    COOLDOWN = 300