from typing import List, Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float = Field(ge=0, le=1000)
    y: float = Field(ge=0, le=1000)


class RoomPolygon(BaseModel):
    id: int
    name: str
    room_type: str = "一般辦公室"
    confidence: float = Field(default=1.0, ge=0, le=1)
    color: str = "#ef4444"
    points: List[Point]

    unit_load: float = 120.0
    per_ping_load: Optional[float] = 650.0
    indoor_model: Optional[str] = None
    indoor_capacity_kw: Optional[float] = None
    outdoor_model: Optional[str] = None
    connection_rate: Optional[float] = None
    included: bool = True


class FloorplanResult(BaseModel):
    rooms: List[RoomPolygon]
