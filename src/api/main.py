from fastapi import FastAPI
import logging
from src.api.routes import router
from src.db.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tfg_domotica")

# Inicializamos la base de datos
init_db()

app = FastAPI(title="TFG Domótica - Detector Anomalías")

# Conectamos nuestras rutas
app.include_router(router)