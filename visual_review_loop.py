from __future__ import annotations

import base64
import io
import json
import os
from copy import deepcopy
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")
NORMALIZED_MAX = 1000.0


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0, le=1000)
    y: float = Field(ge=0, le=1000)


class RoomPolygon(BaseModel):
    id: str
    name: str
    points: list[NormalizedPoint]
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
    position: NormalizedPoint | None = None
    new_name: str | None = None
    reason: str = ""


class ReviewResult(BaseModel):
    overall_status: Literal["approved", "revise"]
    operations: list[ReviewOperation]
    summary: str = ""


INITIAL_PROMPT = """
你是一位建築平面圖空間判讀人員。

你看到的是「已裁切的建築主體」，圖片上覆蓋 0～1000 的標準化座標格線。
請產生主要室內使用空間的初始 Polygon。

座標規則：
1. 所有 x、y 都使用 0～1000 的標準化座標。
2. 左上角是 (0,0)，右下角是 (1000,1000)。
3. 不要使用圖片像素座標。
4. 請參照格線與邊界標籤定位。

空間規則：
1. 框完整房間或完整開放式使用空間。
2. 不框床、桌椅、沙發、櫃體、流理台、洗手台、馬桶、樓梯踏階、電梯設備、門片、窗戶、尺寸線、文字、陽台、庭院、車道或建築外部。
3. 開放式客廳、餐廳、廚房若沒有完整隔牆，視為同一空間。
4. 有完整隔牆的臥室、衛浴、儲藏室分別框選。
5. Polygon 應沿牆內側主要轉折，不得跨越實牆。
6. L 型空間要使用 L 型 Polygon。
7. 每個空間至少 4 個角點，按順時針或逆時針排列。
8. 不確定時降低 confidence，不要假裝精確。
"""


REVIEW_PROMPT = """
你正在執行「視覺修正」，不是重新產生全部房間。

你會看到：
1. 裁切後的原始平面圖與 0～1000 座標格線。
2. 已畫上紅色 Polygon 與藍色角點編號的疊圖。

請逐一檢查：
- 是否框到家具、設備、樓梯、電梯、尺寸線或室外區域。
- 是否穿過牆體。
- 是否漏掉主要轉折。
- 是否應刪除。
- 名稱是否合理。

你只能回傳：
- approve
- move_point
- add_point
- delete_point
- delete_room
- rename_room

重要：
1. 不要重新輸出完整 Polygon。
2. 每次只修正最明顯、最必要的錯誤。
3. 所有位置仍使用 0～1000 標準化座標。
4. 若全部合理，overall_status=approved。
5. 若仍需修正，overall_status=revise。

目前房間 JSON：
{rooms_json}

目前輪次：{round_index}/{max_rounds}
"""


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def clamp_norm(value: float) -> float:
    return max(0.0, min(NORMALIZED_MAX, float(value)))


def norm_to_pixel(point: list[float], width: int, height: int) -> tuple[float, float]:
    x = point[0] / NORMALIZED_MAX * max(width - 1, 1)
    y = point[1] / NORMALIZED_MAX * max(height - 1, 1)
    return x, y


def pixel_to_norm(x: float, y: float, width: int, height: int) -> list[float]:
    nx = x / max(width - 1, 1) * NORMALIZED_MAX
    ny = y / max(height - 1, 1) * NORMALIZED_MAX
    return [clamp_norm(nx), clamp_norm(ny)]


def add_normalized_grid(image: Image.Image, step: int = 100) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    width, height = result.size

    for value in range(0, 1001, step):
        x = round(value / 1000 * (width - 1))
        y = round(value / 1000 * (height - 1))

        line_alpha = 120 if value % 500 == 0 else 65
        line_width = 2 if value % 500 == 0 else 1

        draw.line((x, 0, x, height), fill=(0, 90, 255, line_alpha), width=line_width)
        draw.line((0, y, width, y), fill=(0, 90, 255, line_alpha), width=line_width)

        draw.rectangle((x, 0, min(x + 38, width), 18), fill=(255, 255, 255, 210))
        draw.text((x + 2, 2), str(value), fill=(0, 70, 200, 255))

        draw.rectangle((0, y, 42, min(y + 18, height)), fill=(255, 255, 255, 210))
        draw.text((2, y + 2), str(value), fill=(0, 70, 200, 255))

    return result


def clean_rooms(rooms: list[RoomPolygon]) -> list[dict]:
    cleaned = []
    for room in rooms:
        points = [[clamp_norm(p.x), clamp_norm(p.y)] for p in room.points]
        if len(points) < 3:
            continue
        cleaned.append({
            "id": room.id.strip(),
            "name": room.name.strip(),
            "points": points,
            "confidence": float(room.confidence),
        })
    return cleaned


def detect_initial_rooms(
    gridded_crop: Image.Image,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": INITIAL_PROMPT},
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(gridded_crop),
                    "detail": "high",
                },
            ],
        }],
        text_format=InitialDetection,
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI 未回傳可解析的初始空間資料。")
    return clean_rooms(response.output_parsed.rooms)


def draw_review_overlay(gridded_crop: Image.Image, rooms: list[dict]) -> Image.Image:
    overlay = gridded_crop.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = overlay.size

    for room in rooms:
        pts = [norm_to_pixel(p, width, height) for p in room["points"]]
        if len(pts) < 3:
            continue

        draw.polygon(pts, fill=(255, 0, 0, 35))
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=5, joint="curve")

        for index, (x, y) in enumerate(pts):
            radius = 8
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(0, 100, 255, 255),
                outline=(255, 255, 255, 255),
                width=2,
            )
            draw.text((x + 10, y - 10), f'{room["id"]}:{index}', fill=(0, 70, 220, 255))

        x, y = pts[0]
        draw.text((x + 8, y + 18), f'{room["id"]} {room["name"]}', fill=(255, 0, 0, 255))

    return overlay


def request_review(
    gridded_crop: Image.Image,
    overlay_image: Image.Image,
    rooms: list[dict],
    round_index: int,
    max_rounds: int,
    model: str = DEFAULT_MODEL,
) -> ReviewResult:
    client = OpenAI()
    prompt = REVIEW_PROMPT.format(
        rooms_json=json.dumps(rooms, ensure_ascii=False, indent=2),
        round_index=round_index,
        max_rounds=max_rounds,
    )

    response = client.responses.parse(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(gridded_crop),
                    "detail": "high",
                },
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(overlay_image),
                    "detail": "high",
                },
            ],
        }],
        text_format=ReviewResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI 未回傳可解析的修正指令。")
    return response.output_parsed


def apply_operations(rooms: list[dict], review: ReviewResult) -> list[dict]:
    result = deepcopy(rooms)

    def find_room(room_id: str):
        return next((r for r in result if r["id"] == room_id), None)

    for op in review.operations:
        room = find_room(op.room_id)

        if op.type == "delete_room":
            result = [r for r in result if r["id"] != op.room_id]
            continue

        if room is None:
            continue

        if op.type == "approve":
            continue

        if op.type == "rename_room":
            if op.new_name:
                room["name"] = op.new_name.strip()
            continue

        points = room["points"]

        if op.type == "move_point":
            if (
                op.point_index is not None
                and op.position is not None
                and 0 <= op.point_index < len(points)
            ):
                points[op.point_index] = [
                    clamp_norm(op.position.x),
                    clamp_norm(op.position.y),
                ]

        elif op.type == "add_point":
            if op.position is not None:
                index = len(points) if op.point_index is None else op.point_index
                index = max(0, min(len(points), index))
                points.insert(index, [
                    clamp_norm(op.position.x),
                    clamp_norm(op.position.y),
                ])

        elif op.type == "delete_point":
            if (
                op.point_index is not None
                and len(points) > 3
                and 0 <= op.point_index < len(points)
            ):
                points.pop(op.point_index)

    return result


def polygon_area_pixels(points_norm: list[list[float]], width: int, height: int) -> float:
    pts = [norm_to_pixel(p, width, height) for p in points_norm]
    total = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def crop_norm_to_original_pixel(
    point_norm: list[float],
    crop_box: tuple[int, int, int, int],
) -> list[float]:
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    x = left + point_norm[0] / 1000 * crop_width
    y = top + point_norm[1] / 1000 * crop_height
    return [x, y]


def draw_on_original(
    original: Image.Image,
    rooms: list[dict],
    crop_box: tuple[int, int, int, int],
) -> Image.Image:
    result = original.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")

    for room in rooms:
        pts = [tuple(crop_norm_to_original_pixel(p, crop_box)) for p in room["points"]]
        if len(pts) < 3:
            continue
        draw.polygon(pts, fill=(255, 0, 0, 30))
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=5, joint="curve")
        x, y = pts[0]
        draw.text((x + 8, y + 8), f'{room["id"]} {room["name"]}', fill=(255, 0, 0, 255))

    return result


def run_visual_review_loop(
    original_image: Image.Image,
    crop_box: tuple[int, int, int, int],
    model: str = DEFAULT_MODEL,
    max_rounds: int = 3,
) -> dict:
    original_image = original_image.convert("RGB")
    left, top, right, bottom = crop_box

    if right <= left or bottom <= top:
        raise ValueError("裁切範圍無效。")

    crop = original_image.crop(crop_box)
    gridded_crop = add_normalized_grid(crop)

    rooms = detect_initial_rooms(gridded_crop, model=model)
    history = []

    for round_index in range(1, max_rounds + 1):
        overlay = draw_review_overlay(gridded_crop, rooms)
        review = request_review(
            gridded_crop=gridded_crop,
            overlay_image=overlay,
            rooms=rooms,
            round_index=round_index,
            max_rounds=max_rounds,
            model=model,
        )

        history.append({
            "round": round_index,
            "rooms_before": deepcopy(rooms),
            "review": review.model_dump(),
        })

        if review.overall_status == "approved":
            break

        updated = apply_operations(rooms, review)
        if updated == rooms:
            break
        rooms = updated

    crop_width, crop_height = crop.size
    for room in rooms:
        room["area_pixels_on_crop"] = polygon_area_pixels(
            room["points"],
            crop_width,
            crop_height,
        )
        room["points_original_pixels"] = [
            crop_norm_to_original_pixel(p, crop_box)
            for p in room["points"]
        ]

    final_crop_overlay = draw_review_overlay(gridded_crop, rooms)
    final_original_overlay = draw_on_original(original_image, rooms, crop_box)

    return {
        "rooms": rooms,
        "history": history,
        "crop_box": list(crop_box),
        "crop_size": list(crop.size),
        "gridded_crop": gridded_crop,
        "final_crop_overlay": final_crop_overlay,
        "final_original_overlay": final_original_overlay,
    }
