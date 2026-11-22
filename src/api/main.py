from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from src.api.routes import router
from src.db.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tfg_domotica")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="TFG Domótica - Detector Anomalías", lifespan=lifespan)
app.include_router(router)
