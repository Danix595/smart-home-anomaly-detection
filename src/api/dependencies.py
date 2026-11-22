import secrets
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from src.core import config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    expected = config.SECRET_API_KEY
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API Key no configurada en el servidor. Define API_KEY en el archivo .env.",
        )
    if not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Acceso denegado: API Key incorrecta.")
    return api_key
