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
    points: list[Point]
    unit_load: float = 120.0
    included: bool = True

class FloorplanResult(BaseModel):
    rooms: list[RoomPolygon]
