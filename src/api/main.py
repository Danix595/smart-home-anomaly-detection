from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from src.api.routes import router
from src.db.database import init_db
from src.models.detector import preload_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tfg_domotica")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    preload_model()
    yield


app = FastAPI(title="TFG Domótica - Detector Anomalías", lifespan=lifespan)
app.include_router(router)
