from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


ROOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "object",
            "properties": {
                "is_floor_plan": {"type": "boolean"},
                "quality": {"type": "string", "enum": ["good", "usable", "poor"]},
                "note": {"type": "string"},
            },
            "required": ["is_floor_plan", "quality", "note"],
            "additionalProperties": False,
        },
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "room_name": {"type": "string"},
                    "room_type": {"type": "string"},
                    "include_in_area": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "polygon": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                                "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                            },
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "room_name",
                    "room_type",
                    "include_in_area",
                    "confidence",
                    "polygon",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "excluded_spaces": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
                "additionalProperties": False,
            },
        },
        "overall_note": {"type": "string"},
    },
    "required": ["assessment", "rooms", "excluded_spaces", "overall_note"],
    "additionalProperties": False,
}


@dataclass
class AIRoomDetectionOptions:
    include_balcony: bool = True
    include_corridor: bool = True
    include_stair: bool = False
    include_bathroom: bool = True
    minimum_confidence: float = 0.35
    minimum_polygon_area_ratio: float = 0.001
    maximum_rooms: int = 24
    max_retries: int = 2


def _to_data_url(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _binary_ink(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 11,
    )
    _, global_mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    ink = cv2.bitwise_and(
        adaptive,
        cv2.dilate(global_mask, np.ones((2, 2), np.uint8)),
    )
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def locate_building_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """OpenCV 僅負責找出建築主體，避免基地線與大量留白送入 GPT。"""
    ink = _binary_ink(image)
    height, width = ink.shape
    image_area = float(width * height)

    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (24, 1)),
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 24)),
    )
    structure = cv2.bitwise_or(horizontal, vertical)
    structure = cv2.dilate(
        structure,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=1,
    )
    structure = cv2.morphologyEx(
        structure, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
    )

    count, _, stats, centroids = cv2.connectedComponentsWithStats(structure, 8)
    image_cx, image_cy = width / 2, height / 2
    best_box = None
    best_score = -1.0

    for label in range(1, count):
        x, y, box_w, box_h, area = stats[label]
        box_area = float(box_w * box_h)
        ratio = box_area / max(image_area, 1.0)
        if ratio < 0.025 or ratio > 0.88 or min(box_w, box_h) < 90:
            continue

        density = (
            np.count_nonzero(ink[y:y + box_h, x:x + box_w])
            / max(box_area, 1.0)
        )
        cx, cy = centroids[label]
        distance = math.hypot(
            (cx - image_cx) / max(width, 1),
            (cy - image_cy) / max(height, 1),
        )
        center_weight = max(0.25, 1.0 - distance)
        aspect = max(box_w / max(box_h, 1), box_h / max(box_w, 1))
        aspect_weight = 1.0 / max(1.0, aspect / 3.5)
        score = area * max(density, 0.004) * center_weight * aspect_weight

        if score > best_score:
            best_score = score
            best_box = (int(x), int(y), int(box_w), int(box_h))

    if best_box is None:
        ys, xs = np.where(ink > 0)
        if len(xs) == 0:
            return 0, 0, width, height
        best_box = (
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )

    x, y, box_w, box_h = best_box
    padding = max(16, round(min(box_w, box_h) * 0.05))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + box_w + padding)
    y1 = min(height, y + box_h + padding)
    return x0, y0, x1 - x0, y1 - y0


def draw_building_preview(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    x, y, w, h = box
    draw.rectangle((x, y, x + w, y + h), outline="#EF4444", width=4)
    return output


def _coordinate_grid(image: Image.Image) -> Image.Image:
    """加入 0～1000 座標網格，提升模型回傳座標的一致性。"""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size

    for index in range(11):
        x = round(width * index / 10)
        y = round(height * index / 10)
        draw.line((x, 0, x, height), fill=(60, 120, 220), width=1)
        draw.line((0, y, width, y), fill=(60, 120, 220), width=1)
        if index < 10:
            draw.text((x + 3, 3), str(index * 100), fill=(20, 70, 160))
            draw.text((3, y + 3), str(index * 100), fill=(20, 70, 160))

    return output


def _repair_polygon(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return []

    bounded = [
        (
            min(max(float(x), 0.0), width - 1.0),
            min(max(float(y), 0.0), height - 1.0),
        )
        for x, y in points
    ]

    polygon = Polygon(bounded)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return []
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)

    polygon = polygon.simplify(
        max(width, height) * 0.001,
        preserve_topology=True,
    )
    return [
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]


def _include_by_type(
    room: dict[str, Any],
    options: AIRoomDetectionOptions,
) -> bool:
    text = (
        f"{room.get('room_name', '')} "
        f"{room.get('room_type', '')}"
    ).lower()

    if any(term in text for term in ["樓梯", "梯間", "stair"]):
        return options.include_stair
    if any(term in text for term in ["陽台", "露台", "balcony"]):
        return options.include_balcony
    if any(term in text for term in ["走道", "走廊", "玄關", "corridor"]):
        return options.include_corridor
    if any(term in text for term in ["衛浴", "浴室", "廁所", "bathroom", "toilet"]):
        return options.include_bathroom
    return bool(room.get("include_in_area", True))


def _deduplicate_rooms(
    rooms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rooms,
        key=lambda room: (
            float(room["confidence"]),
            float(room["area_px2"]),
        ),
        reverse=True,
    )
    accepted: list[dict[str, Any]] = []

    for candidate in ordered:
        candidate_polygon = Polygon(candidate["points"])
        duplicate = False

        for existing in accepted:
            existing_polygon = Polygon(existing["points"])
            union = candidate_polygon.union(existing_polygon).area
            iou = (
                candidate_polygon.intersection(existing_polygon).area / union
                if union > 0 else 0
            )
            if iou >= 0.55:
                duplicate = True
                break

        if not duplicate:
            accepted.append(candidate)

    accepted.sort(
        key=lambda room: (
            min(y for _, y in room["points"]),
            min(x for x, _ in room["points"]),
        )
    )
    for index, room in enumerate(accepted, start=1):
        room["room_id"] = f"R{index:02d}"
    return accepted


def detect_rooms_with_openai(
    api_key: str,
    image: Image.Image,
    model: str = "gpt-4.1",
    options: AIRoomDetectionOptions | None = None,
) -> dict[str, Any]:
    """OpenCV 找建築主體，GPT 針對完整建築裁切圖辨識房間。

    不再把建築切成多個區塊，避免跨區房間被放大、重複或錯誤拼接。
    """
    if not api_key:
        raise ValueError("尚未設定 OPENAI_API_KEY。")

    options = options or AIRoomDetectionOptions()
    source = image.convert("RGB")
    box = locate_building_bbox(source)
    box_x, box_y, box_w, box_h = box
    building = source.crop(
        (box_x, box_y, box_x + box_w, box_y + box_h)
    )
    grid = _coordinate_grid(building)

    prompt = f"""
你是建築平面圖空間辨識助手。OpenCV 已經先裁切出建築主體；
請在這個完整建築裁切圖中辨識所有由牆體圍合的室內空間。

你會收到兩張相同內容的圖片：
1. 原始建築裁切圖。
2. 加上 0～1000 座標網格的輔助圖。
所有 polygon 座標都以裁切圖為準，左上角 (0,0)，右下角 (1000,1000)。

重要規則：
- 先理解整棟平面配置，再逐一輸出空間。
- polygon 要貼近「內牆完成面」，不能使用外接大矩形。
- L 型、凹型或轉折空間要保留轉折點。
- 門洞處沿牆面方向補成房間邊界。
- 忽略尺寸線、尺寸數字、家具、床、桌椅、櫃體分格、門片開啟弧、
  樓梯踏階、窗框內線及基地界線。
- 開放式客餐廳沒有牆體分隔時，視為同一空間。
- 同一房間只輸出一次，不可產生跨越多個房間的大框。
- 不可把整層樓、尺寸區或家具區當成單一房間。
- 小型衛浴、儲藏室、走道也要檢查。
- 最少 4 點、最多 24 點；不確定時降低 confidence，不要虛構。

納入面積：
陽台={"是" if options.include_balcony else "否"}；
走道／玄關={"是" if options.include_corridor else "否"}；
衛浴={"是" if options.include_bathroom else "否"}；
樓梯={"是" if options.include_stair else "否"}。

最多輸出 {options.maximum_rooms} 個空間。
這些是可人工修改的候選框，不得宣稱為 CAD 或正式測量精度。
"""

    client = OpenAI(api_key=api_key)
    response = None
    last_error: Exception | None = None

    for attempt in range(options.max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": _to_data_url(building),
                                "detail": "high",
                            },
                            {
                                "type": "input_image",
                                "image_url": _to_data_url(grid),
                                "detail": "high",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "floorplan_rooms",
                        "strict": True,
                        "schema": ROOM_SCHEMA,
                    }
                },
            )
            break
        except Exception as error:
            last_error = error
            if attempt >= options.max_retries:
                raise
            time.sleep(1.5 * (attempt + 1))

    if response is None:
        raise RuntimeError(str(last_error or "OpenAI 未回傳結果"))

    raw = json.loads(response.output_text)
    image_area = float(source.width * source.height)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, room in enumerate(raw.get("rooms", []), start=1):
        local_points = [
            (
                float(point["x"]) / 1000.0 * box_w,
                float(point["y"]) / 1000.0 * box_h,
            )
            for point in room.get("polygon", [])
        ]
        global_points = _repair_polygon(
            [(box_x + x, box_y + y) for x, y in local_points],
            source.width,
            source.height,
        )
        area_px2 = (
            float(Polygon(global_points).area)
            if len(global_points) >= 3 else 0.0
        )
        confidence = float(room.get("confidence", 0.0))

        reasons: list[str] = []
        if confidence < options.minimum_confidence:
            reasons.append("低於最低信心分數")
        if len(global_points) < 3:
            reasons.append("多邊形無效")
        if area_px2 < image_area * options.minimum_polygon_area_ratio:
            reasons.append("面積過小")
        if area_px2 > image_area * 0.42:
            reasons.append("面積異常過大，疑似跨越多個房間")

        record = {
            "room_id": f"R{index:02d}",
            "room_name": room.get("room_name") or "未命名空間",
            "room_type": room.get("room_type") or "無法判斷",
            "include_in_area": _include_by_type(room, options),
            "confidence": confidence,
            "points": global_points,
            "area_px2": area_px2,
            "reason": room.get("reason", ""),
            "source": "opencv_crop_gpt",
        }

        if reasons:
            record["rejected_reason"] = "、".join(reasons)
            rejected.append(record)
        else:
            candidates.append(record)

    rooms = _deduplicate_rooms(candidates)

    return {
        "rooms": rooms,
        "rejected_rooms": rejected,
        "excluded_spaces": raw.get("excluded_spaces", []),
        "image_assessment": {
            "is_floor_plan": raw.get("assessment", {}).get("is_floor_plan", True),
            "building_region_found": True,
            "quality": raw.get("assessment", {}).get("quality", "usable"),
            "note": raw.get("assessment", {}).get("note", ""),
        },
        "overall_note": raw.get("overall_note", ""),
        "model": model,
        "image_width": source.width,
        "image_height": source.height,
        "building_box": box,
    }
