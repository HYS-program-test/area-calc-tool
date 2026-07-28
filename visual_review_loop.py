from __future__ import annotations

import base64
import io
import json
import math
import os
from copy import deepcopy
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")


class Point(BaseModel):
    x: float
    y: float


class RoomPolygon(BaseModel):
    id: str
    name: str
    points: list[Point]
    confidence: float = Field(ge=0.0, le=1.0)


class InitialDetection(BaseModel):
    rooms: list[RoomPolygon]


class ReviewOperation(BaseModel):
    type: Literal[
        "approve",
        "move_point",
        "add_point",
        "delete_point",
        "delete_room",
        "rename_room",
    ]
    room_id: str
    point_index: int | None = None
    position: Point | None = None
    new_name: str | None = None
    reason: str = ""


class ReviewResult(BaseModel):
    overall_status: Literal["approved", "revise"]
    operations: list[ReviewOperation]
    summary: str = ""


INITIAL_PROMPT = """
你是一位建築平面圖空間判讀人員。

請閱讀輸入圖面，先產生主要室內使用空間的初始 Polygon。
這是候選結果，後續會再把疊圖交給你修正，因此請優先完整辨識空間，不要輸出家具物件。

規則：
1. 框完整房間或完整開放式使用空間。
2. 不框床、桌椅、沙發、櫃體、流理台、洗手台、馬桶、樓梯踏階、電梯設備、門片、窗戶、尺寸線、文字、陽台、庭院、車道或建築外部。
3. 開放式客廳、餐廳、廚房若沒有完整隔牆，視為同一空間。
4. 有完整隔牆的臥室、衛浴、儲藏室分別框選。
5. Polygon 應沿牆內側主要轉折，不得跨越實牆。
6. L 型空間要使用 L 型 Polygon。
7. 所有座標使用輸入圖片的實際像素座標。
8. 每個空間至少 4 個角點，角點按順時針或逆時針排列。
9. 不確定時降低 confidence，不要假裝精確。

輸入圖片尺寸：width={width}, height={height}
"""


REVIEW_PROMPT = """
你正在執行「視覺修正」，不是重新產生全部房間。

你會同時看到：
1. 原始平面圖。
2. 已畫上紅色 Polygon 與藍色角點編號的疊圖。

請逐一檢查每個房間：
- 是否框到家具、設備、樓梯、電梯、尺寸線或室外區域。
- 是否穿過牆體。
- 是否漏掉主要轉折。
- 是否應刪除。
- 房間名稱是否合理。

你只能回傳有限修改命令：
- approve：該房間不需修改。
- move_point：移動既有角點。
- add_point：在指定 point_index 之前插入角點。
- delete_point：刪除角點。
- delete_room：刪除整個房間。
- rename_room：修改名稱。

重要：
- 不要重新輸出完整 Polygon。
- 每次只修正最明顯、最必要的錯誤。
- 座標使用原始圖片像素座標。
- 若全部已合理，overall_status=approved。
- 若仍需修正，overall_status=revise。

目前房間 JSON：
{rooms_json}

目前輪次：{round_index}/{max_rounds}
"""


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _clamp(value: float, maximum: int) -> float:
    return max(0.0, min(float(maximum - 1), float(value)))


def _clean_rooms(
    rooms: list[RoomPolygon],
    width: int,
    height: int,
) -> list[dict]:
    cleaned: list[dict] = []

    for room in rooms:
        points = [
            [_clamp(point.x, width), _clamp(point.y, height)]
            for point in room.points
        ]

        if len(points) < 3:
            continue

        cleaned.append(
            {
                "id": room.id,
                "name": room.name,
                "points": points,
                "confidence": float(room.confidence),
            }
        )

    return cleaned


def detect_initial_rooms(
    image: Image.Image,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    client = OpenAI()
    width, height = image.size

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": INITIAL_PROMPT.format(
                            width=width,
                            height=height,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(image),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=InitialDetection,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI 未回傳可解析的初始空間資料。")

    return _clean_rooms(parsed.rooms, width, height)


def draw_review_overlay(
    image: Image.Image,
    rooms: list[dict],
) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay, "RGBA")

    for room in rooms:
        points = [tuple(point) for point in room["points"]]
        if len(points) < 3:
            continue

        draw.polygon(points, fill=(255, 0, 0, 35))
        draw.line(
            points + [points[0]],
            fill=(255, 0, 0, 255),
            width=5,
            joint="curve",
        )

        for index, (x, y) in enumerate(points):
            radius = 8
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(0, 90, 255, 255),
                outline=(255, 255, 255, 255),
                width=2,
            )
            draw.text(
                (x + 10, y - 10),
                f'{room["id"]}:{index}',
                fill=(0, 60, 220, 255),
            )

        x, y = points[0]
        draw.text(
            (x + 8, y + 18),
            f'{room["id"]} {room["name"]}',
            fill=(255, 0, 0, 255),
        )

    return overlay


def request_review(
    original_image: Image.Image,
    overlay_image: Image.Image,
    rooms: list[dict],
    round_index: int,
    max_rounds: int,
    model: str = DEFAULT_MODEL,
) -> ReviewResult:
    client = OpenAI()

    prompt = REVIEW_PROMPT.format(
        rooms_json=json.dumps(
            rooms,
            ensure_ascii=False,
            indent=2,
        ),
        round_index=round_index,
        max_rounds=max_rounds,
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(original_image),
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(overlay_image),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=ReviewResult,
    )

    if response.output_parsed is None:
        raise RuntimeError("OpenAI 未回傳可解析的修正指令。")

    return response.output_parsed


def apply_operations(
    rooms: list[dict],
    review: ReviewResult,
    width: int,
    height: int,
) -> list[dict]:
    result = deepcopy(rooms)

    def find_room(room_id: str) -> dict | None:
        return next(
            (room for room in result if room["id"] == room_id),
            None,
        )

    for operation in review.operations:
        room = find_room(operation.room_id)

        if operation.type == "delete_room":
            result = [
                item for item in result
                if item["id"] != operation.room_id
            ]
            continue

        if room is None:
            continue

        if operation.type == "approve":
            continue

        if operation.type == "rename_room":
            if operation.new_name:
                room["name"] = operation.new_name.strip()
            continue

        points = room["points"]

        if operation.type == "move_point":
            if (
                operation.point_index is not None
                and operation.position is not None
                and 0 <= operation.point_index < len(points)
            ):
                points[operation.point_index] = [
                    _clamp(operation.position.x, width),
                    _clamp(operation.position.y, height),
                ]

        elif operation.type == "add_point":
            if operation.position is not None:
                index = operation.point_index
                if index is None:
                    index = len(points)
                index = max(0, min(len(points), index))
                points.insert(
                    index,
                    [
                        _clamp(operation.position.x, width),
                        _clamp(operation.position.y, height),
                    ],
                )

        elif operation.type == "delete_point":
            if (
                operation.point_index is not None
                and len(points) > 3
                and 0 <= operation.point_index < len(points)
            ):
                points.pop(operation.point_index)

    return result


def polygon_area_pixels(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0

    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1

    return abs(total) / 2.0


def run_visual_review_loop(
    image: Image.Image,
    model: str = DEFAULT_MODEL,
    max_rounds: int = 3,
) -> dict:
    image = image.convert("RGB")
    width, height = image.size

    rooms = detect_initial_rooms(image, model=model)
    history: list[dict] = []

    for round_index in range(1, max_rounds + 1):
        overlay = draw_review_overlay(image, rooms)
        review = request_review(
            original_image=image,
            overlay_image=overlay,
            rooms=rooms,
            round_index=round_index,
            max_rounds=max_rounds,
            model=model,
        )

        history.append(
            {
                "round": round_index,
                "rooms_before": deepcopy(rooms),
                "review": review.model_dump(),
            }
        )

        if review.overall_status == "approved":
            break

        updated = apply_operations(
            rooms,
            review,
            width,
            height,
        )

        if updated == rooms:
            break

        rooms = updated

    final_overlay = draw_review_overlay(image, rooms)

    for room in rooms:
        room["area_pixels"] = polygon_area_pixels(room["points"])

    return {
        "rooms": rooms,
        "history": history,
        "final_overlay": final_overlay,
    }
