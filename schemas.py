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

    # 這兩個欄位不是給計算用的，是刻意讓模型「先講出判斷依據」再給座標——
    # 逼模型多想一步，通常能讓多邊形描繪更準（尤其是不規則形狀的空間）。
    label_on_plan: Optional[str] = None   # 圖上如果有印房間名稱文字，原樣填入；沒有就留空
    wall_trace_notes: Optional[str] = None  # 一句話說明這個空間的邊界怎麼判斷出來的

    unit_load: float = 120.0
    per_ping_load: Optional[float] = 650.0

    indoor_model: Optional[str] = None
    indoor_quantity: int = 1
    indoor_capacity_kw: Optional[float] = None
    outdoor_model: Optional[str] = None
    connection_rate: Optional[float] = None

    included: bool = True


class FloorplanResult(BaseModel):
    rooms: List[RoomPolygon]
    room_count_check: Optional[str] = None  # 一句話自我檢查：數出的空間數量是否合理、有無明顯遺漏

