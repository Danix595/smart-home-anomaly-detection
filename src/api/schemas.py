from pydantic import BaseModel
from typing import Optional, Dict

class EnergyData(BaseModel):
    global_power_W: Optional[float] = None
    plugs_W: Optional[Dict[str, float]] = None  # Ej: {"enchufe_tv": 50.0, "enchufe_lavadora": 0.0}
    lights_on: Optional[Dict[str, bool]] = None # Ej: {"luz_salon": True}

class SecurityData(BaseModel):
    doors_open: Optional[Dict[str, bool]] = None   # Ej: {"puerta_principal": False}
    windows_open: Optional[Dict[str, bool]] = None # Ej: {"ventana_bano": True}

class ComfortData(BaseModel):
    heating_on: Optional[Dict[str, bool]] = None       # Ej: {"termostato_salon": True}
    target_temp: Optional[Dict[str, float]] = None     # Ej: {"termostato_salon": 21.5}
    blinds_open: Optional[Dict[str, bool]] = None      # Ej: {"persiana_habitacion": False}

class HealthData(BaseModel):
    presence: Optional[int] = None          # 1 = En casa, 0 = Fuera
    motion: Optional[int] = None            # 1 = Movimiento, 0 = Sin movimiento
    fp2_zones: Optional[Dict[str, bool]] = None # Ej: {"sofa": True, "cama": False}

class WaterData(BaseModel):
    water_L_per_min: Optional[float] = None

class Reading(BaseModel):
    timestamp: Optional[str] = None
    energy: Optional[EnergyData] = None
    security: Optional[SecurityData] = None
    comfort: Optional[ComfortData] = None
    health: Optional[HealthData] = None
    water: Optional[WaterData] = None