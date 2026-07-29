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

    # 舊版欄位保留，讓既有 OpenAI 回傳格式仍可使用。
    unit_load: float = 120.0

    # 新版空調負荷與設備選型欄位。
    per_ping_load: float | None = None
    indoor_model: str | None = None
    indoor_capacity_kw: float | None = None
    outdoor_model: str | None = None
    outdoor_capacity_kw: float | None = None

    included: bool = True


class FloorplanResult(BaseModel):
    rooms: list[RoomPolygon]
