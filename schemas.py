from pydantic import BaseModel, Field

class Point(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)

class RoomPolygon(BaseModel):
    name: str
    room_type: str
    confidence: float = Field(ge=0, le=1)
    points: list[Point]
    notes: str = ""

class FloorplanResult(BaseModel):
    rooms: list[RoomPolygon]
