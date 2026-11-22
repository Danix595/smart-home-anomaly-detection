from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from src.core.config import SECRET_API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Acceso denegado: API Key incorrecta.")
    return api_key