from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator


class EnergyData(BaseModel):
    global_power_W: Optional[float] = Field(default=None, ge=0.0)
    plugs_W: Optional[Dict[str, float]] = None  # Ej: {"enchufe_tv": 50.0, "enchufe_lavadora": 0.0}
    lights_on: Optional[Dict[str, bool]] = None  # Ej: {"luz_salon": True}

    @field_validator("plugs_W")
    @classmethod
    def validate_plugs_non_negative(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if value is not None:
            for name, watts in value.items():
                if watts < 0:
                    raise ValueError(f"La potencia del enchufe '{name}' no puede ser negativa ({watts}).")
        return value


class SecurityData(BaseModel):
    doors_open: Optional[Dict[str, bool]] = None    # Ej: {"puerta_principal": False}
    windows_open: Optional[Dict[str, bool]] = None  # Ej: {"ventana_bano": True}


class ComfortData(BaseModel):
    heating_on: Optional[Dict[str, bool]] = None    # Ej: {"termostato_salon": True}
    target_temp: Optional[Dict[str, float]] = None  # Ej: {"termostato_salon": 21.5}
    blinds_open: Optional[Dict[str, bool]] = None   # Ej: {"persiana_habitacion": False}

    @field_validator("target_temp")
    @classmethod
    def validate_target_temp_range(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if value is not None:
            for name, temp in value.items():
                if not (0.0 <= temp <= 45.0):
                    raise ValueError(
                        f"La temperatura objetivo de '{name}' ({temp} °C) está fuera de rango (0–45 °C)."
                    )
        return value


class HealthData(BaseModel):
    presence: Optional[int] = Field(default=None, ge=0, le=1)  # 1 = En casa, 0 = Fuera
    motion: Optional[int] = Field(default=None, ge=0, le=1)    # 1 = Movimiento, 0 = Sin movimiento
    fp2_zones: Optional[Dict[str, bool]] = None                # Ej: {"sofa": True, "cama": False}


class WaterData(BaseModel):
    water_L_per_min: Optional[float] = Field(default=None, ge=0.0)


class Reading(BaseModel):
    timestamp: Optional[str] = None
    energy: Optional[EnergyData] = None
    security: Optional[SecurityData] = None
    comfort: Optional[ComfortData] = None
    health: Optional[HealthData] = None
    water: Optional[WaterData] = None

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Timestamp ISO-8601 inválido: '{value}'.") from exc
        return value
