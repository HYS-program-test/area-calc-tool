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
from PIL import Image
from shapely.geometry import Polygon


GLOBAL_SCHEMA: dict[str, Any] = {
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
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x1": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "y1": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "x2": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "y2": {"type": "integer", "minimum": 0, "maximum": 1000},
                        },
                        "required": ["x1", "y1", "x2", "y2"],
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "room_name",
                    "room_type",
                    "include_in_area",
                    "confidence",
                    "bbox",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assessment", "rooms"],
    "additionalProperties": False,
}


LOCAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valid_room": {"type": "boolean"},
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
        "note": {"type": "string"},
    },
    "required": ["valid_room", "confidence", "polygon", "note"],
    "additionalProperties": False,
}


@dataclass
class AIRoomDetectionOptions:
    include_balcony: bool = True
    include_corridor: bool = True
    include_stair: bool = False
    include_bathroom: bool = True
    minimum_confidence: float = 0.40
    minimum_polygon_area_ratio: float = 0.0008
    maximum_rooms: int = 20
    crop_context_ratio: float = 0.16
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
    return cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )


def locate_building_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """OpenCV 只定位完整建築主體，不直接判斷房間。"""
    ink = _binary_ink(image)
    height, width = ink.shape
    image_area = float(width * height)

    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (24, 1)),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 24)),
    )
    structure = cv2.bitwise_or(horizontal, vertical)
    structure = cv2.dilate(
        structure,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=1,
    )
    structure = cv2.morphologyEx(
        structure,
        cv2.MORPH_CLOSE,
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
        if ratio < 0.025 or ratio > 0.90 or min(box_w, box_h) < 90:
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
            int(xs.min()),
            int(ys.min()),
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


def _request_with_retry(
    client: OpenAI,
    model: str,
    input_payload: list[dict[str, Any]],
    schema_name: str,
    schema: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=input_payload,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            return json.loads(response.output_text)
        except Exception as error:
            last_error = error
            if attempt >= retries:
                raise
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(str(last_error or "OpenAI 未回傳結果"))


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


def _repair_polygon(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return []

    points = [
        (
            min(max(float(x), 0.0), width - 1.0),
            min(max(float(y), 0.0), height - 1.0),
        )
        for x, y in points
    ]

    polygon = Polygon(points)
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
    if polygon.is_empty:
        return []

    return [
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]


def _bbox_to_pixels(
    bbox: dict[str, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = round(float(bbox["x1"]) / 1000 * width)
    y1 = round(float(bbox["y1"]) / 1000 * height)
    x2 = round(float(bbox["x2"]) / 1000 * width)
    y2 = round(float(bbox["y2"]) / 1000 * height)

    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _expanded_crop(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x, y, box_w, box_h = box
    margin_x = round(box_w * ratio)
    margin_y = round(box_h * ratio)

    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(width, x + box_w + margin_x)
    y1 = min(height, y + box_h + margin_y)
    return x0, y0, x1 - x0, y1 - y0


def _iou(first: Polygon, second: Polygon) -> float:
    union = first.union(second).area
    return 0.0 if union <= 0 else float(first.intersection(second).area / union)


def _deduplicate(
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
        if any(
            _iou(candidate_polygon, Polygon(existing["points"])) >= 0.50
            for existing in accepted
        ):
            continue
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
    """兩階段辨識：全圖找空間，再逐房間局部精修邊界。"""
    if not api_key:
        raise ValueError("尚未設定 OPENAI_API_KEY。")

    options = options or AIRoomDetectionOptions()
    source = image.convert("RGB")
    building_box = locate_building_bbox(source)
    bx, by, bw, bh = building_box
    building = source.crop((bx, by, bx + bw, by + bh))

    client = OpenAI(api_key=api_key)

    global_prompt = f"""
你是建築平面圖空間辨識助手。請先理解完整建築配置，
只回傳每個空間的大致 bounding box，不要在此階段回傳 polygon。

規則：
- 忽略尺寸線、尺寸數字、家具、床、桌椅、櫃體分格、門片開啟弧、
  樓梯踏階、窗框內線及基地界線。
- 每個 bbox 只包住一個空間，不能跨越多個房間。
- 開放式客餐廳若沒有牆體分隔，視為同一空間。
- 小型衛浴、儲藏室、走道也要檢查。
- bbox 使用 0～1000 座標，左上角為 (0,0)，右下角為 (1000,1000)。
- 不確定時降低 confidence，不要虛構。

納入面積：
陽台={"是" if options.include_balcony else "否"}；
走道／玄關={"是" if options.include_corridor else "否"}；
衛浴={"是" if options.include_bathroom else "否"}；
樓梯={"是" if options.include_stair else "否"}。

最多輸出 {options.maximum_rooms} 個空間。
"""

    global_result = _request_with_retry(
        client,
        model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": global_prompt},
                    {
                        "type": "input_image",
                        "image_url": _to_data_url(building),
                        "detail": "high",
                    },
                ],
            }
        ],
        "floorplan_global_rooms",
        GLOBAL_SCHEMA,
        options.max_retries,
    )

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    image_area = float(source.width * source.height)

    for global_room in global_result.get("rooms", []):
        global_confidence = float(global_room.get("confidence", 0))
        if global_confidence < options.minimum_confidence:
            rejected.append(
                {
                    "room_name": global_room.get("room_name", "未命名"),
                    "confidence": global_confidence,
                    "rejected_reason": "全圖辨識信心分數過低",
                }
            )
            continue

        local_box = _bbox_to_pixels(
            global_room["bbox"],
            building.width,
            building.height,
        )
        crop_box = _expanded_crop(
            local_box,
            building.width,
            building.height,
            options.crop_context_ratio,
        )
        cx, cy, cw, ch = crop_box
        crop = building.crop((cx, cy, cx + cw, cy + ch))

        local_prompt = f"""
這是一個完整平面圖中「{global_room.get("room_name", "未命名空間")}」
附近的局部裁切圖。請只找出這一個指定空間的內牆完成面邊界。

規則：
- polygon 使用這張局部圖的 0～1000 座標。
- 只輸出指定空間，不要框相鄰房間。
- 忽略家具、尺寸線、文字、門片弧線、樓梯踏階及窗框內線。
- 門洞處沿牆面方向補成房間邊界。
- L 型或凹型空間必須保留轉折，不能使用外接矩形。
- 若局部圖無法確認是有效空間，valid_room=false。
- 這是候選框，不得宣稱 CAD 或正式測量精度。
"""

        local_result = _request_with_retry(
            client,
            model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": local_prompt},
                        {
                            "type": "input_image",
                            "image_url": _to_data_url(crop),
                            "detail": "high",
                        },
                    ],
                }
            ],
            "floorplan_local_polygon",
            LOCAL_SCHEMA,
            options.max_retries,
        )

        local_confidence = float(local_result.get("confidence", 0))
        confidence = min(global_confidence, local_confidence)

        if not local_result.get("valid_room", False):
            rejected.append(
                {
                    "room_name": global_room.get("room_name", "未命名"),
                    "confidence": confidence,
                    "rejected_reason": "局部精修判定不是有效房間",
                }
            )
            continue

        local_points = [
            (
                float(point["x"]) / 1000 * cw,
                float(point["y"]) / 1000 * ch,
            )
            for point in local_result.get("polygon", [])
        ]

        full_points = _repair_polygon(
            [
                (
                    bx + cx + x,
                    by + cy + y,
                )
                for x, y in local_points
            ],
            source.width,
            source.height,
        )

        area_px2 = (
            float(Polygon(full_points).area)
            if len(full_points) >= 3
            else 0.0
        )

        reasons: list[str] = []
        if confidence < options.minimum_confidence:
            reasons.append("局部精修信心分數過低")
        if len(full_points) < 3:
            reasons.append("多邊形無效")
        if area_px2 < image_area * options.minimum_polygon_area_ratio:
            reasons.append("候選面積過小")
        if area_px2 > image_area * 0.30:
            reasons.append("候選面積異常過大")

        record = {
            "room_id": "",
            "room_name": global_room.get("room_name") or "未命名空間",
            "room_type": global_room.get("room_type") or "無法判斷",
            "include_in_area": _include_by_type(global_room, options),
            "confidence": confidence,
            "points": full_points,
            "area_px2": area_px2,
            "reason": (
                f"{global_room.get('reason', '')}；"
                f"{local_result.get('note', '')}"
            ),
            "source": "gpt_global_local",
        }

        if reasons:
            record["rejected_reason"] = "、".join(reasons)
            rejected.append(record)
        else:
            candidates.append(record)

    rooms = _deduplicate(candidates)

    return {
        "rooms": rooms,
        "rejected_rooms": rejected,
        "excluded_spaces": [],
        "image_assessment": {
            "is_floor_plan": global_result.get("assessment", {}).get(
                "is_floor_plan", True
            ),
            "building_region_found": True,
            "quality": global_result.get("assessment", {}).get(
                "quality", "usable"
            ),
            "note": global_result.get("assessment", {}).get("note", ""),
        },
        "overall_note": (
            "先以完整圖面辨識空間，再逐房間局部精修邊界。"
            "結果仍須人工確認。"
        ),
        "model": model,
        "image_width": source.width,
        "image_height": source.height,
        "building_box": building_box,
    }
