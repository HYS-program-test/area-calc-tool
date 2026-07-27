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


@dataclass
class AIRoomDetectionOptions:
    include_balcony: bool = True
    include_corridor: bool = True
    include_stair: bool = False
    include_bathroom: bool = True
    minimum_confidence: float = 0.42
    minimum_polygon_area_ratio: float = 0.0008
    maximum_rooms: int = 20
    max_retries: int = 2


def _to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _request(
    client: OpenAI,
    model: str,
    prompt: str,
    image: Image.Image,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
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
                                "image_url": _to_data_url(image),
                                "detail": "high",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "floorplan_room_boxes",
                        "strict": True,
                        "schema": GLOBAL_SCHEMA,
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


def _binary_line_mask(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    _, dark = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    ink = cv2.bitwise_and(
        adaptive,
        cv2.dilate(dark, np.ones((2, 2), np.uint8)),
    )

    # 強化主要水平／垂直牆線，弱化文字與家具曲線。
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)),
    )
    structural = cv2.bitwise_or(horizontal, vertical)

    near_structure = cv2.dilate(
        structural,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    walls = cv2.bitwise_and(ink, near_structure)
    walls = cv2.bitwise_or(walls, structural)
    walls = cv2.dilate(walls, np.ones((2, 2), np.uint8), iterations=1)

    # 只補小門洞，不把走道及相鄰房間整片封死。
    walls_h = cv2.morphologyEx(
        walls,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
    )
    walls_v = cv2.morphologyEx(
        walls,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
    )
    return cv2.bitwise_or(walls_h, walls_v)


def _bbox_to_pixels(
    bbox: dict[str, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = round(float(bbox["x1"]) / 1000 * width)
    y1 = round(float(bbox["y1"]) / 1000 * height)
    x2 = round(float(bbox["x2"]) / 1000 * width)
    y2 = round(float(bbox["y2"]) / 1000 * height)

    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    return x1, y1, max(2, x2 - x1), max(2, y2 - y1)


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
        max(width, height) * 0.0015,
        preserve_topology=True,
    )
    if polygon.is_empty:
        return []

    return [
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]


def _seed_points(
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    x, y, w, h = bbox
    fractions = [
        (0.50, 0.50),
        (0.40, 0.50),
        (0.60, 0.50),
        (0.50, 0.40),
        (0.50, 0.60),
        (0.35, 0.35),
        (0.65, 0.35),
        (0.35, 0.65),
        (0.65, 0.65),
    ]
    return [
        (round(x + w * fx), round(y + h * fy))
        for fx, fy in fractions
    ]


def _component_from_seed(
    free_space: np.ndarray,
    seed: tuple[int, int],
) -> np.ndarray | None:
    x, y = seed
    height, width = free_space.shape
    if not (0 <= x < width and 0 <= y < height):
        return None

    # 尋找 seed 周圍最近的空白像素。
    if free_space[y, x] == 0:
        found = None
        for radius in range(2, 18, 2):
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            ys, xs = np.where(free_space[y0:y1, x0:x1] > 0)
            if len(xs):
                found = (x0 + int(xs[0]), y0 + int(ys[0]))
                break
        if found is None:
            return None
        x, y = found

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        free_space,
        8,
    )
    label = int(labels[y, x])
    if label <= 0 or label >= count:
        return None

    component = np.zeros_like(free_space)
    component[labels == label] = 255
    return component


def _local_boundary_from_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    """用 GPT bbox 當 seed，OpenCV 只在局部找封閉室內空白區域。"""
    walls = _binary_line_mask(image)
    free_space = cv2.bitwise_not(walls)

    image_height, image_width = free_space.shape
    x, y, w, h = bbox

    # 限制搜尋範圍，避免 seed 漏到整張圖的背景。
    margin_x = round(w * 0.22)
    margin_y = round(h * 0.22)
    rx0 = max(0, x - margin_x)
    ry0 = max(0, y - margin_y)
    rx1 = min(image_width, x + w + margin_x)
    ry1 = min(image_height, y + h + margin_y)

    local_mask = np.zeros_like(free_space)
    local_mask[ry0:ry1, rx0:rx1] = 255
    constrained = cv2.bitwise_and(free_space, local_mask)

    bbox_polygon = Polygon([
        (x, y), (x + w, y), (x + w, y + h), (x, y + h),
    ])

    best_points: list[tuple[float, float]] = []
    best_score = -1.0

    for seed in _seed_points(bbox):
        component = _component_from_seed(constrained, seed)
        if component is None:
            continue

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < max(250, w * h * 0.12):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour,
            max(1.5, perimeter * 0.006),
            True,
        )
        points = [
            (float(item[0][0]), float(item[0][1]))
            for item in approx
        ]
        points = _repair_polygon(points, image_width, image_height)
        if len(points) < 3:
            continue

        polygon = Polygon(points)
        intersection = polygon.intersection(bbox_polygon).area
        coverage = intersection / max(bbox_polygon.area, 1.0)
        size_ratio = polygon.area / max(bbox_polygon.area, 1.0)

        # 偏好覆蓋 GPT bbox 主要區域、但不無限擴大的候選。
        oversize_penalty = max(0.0, size_ratio - 1.8)
        score = coverage - oversize_penalty * 0.45

        if score > best_score:
            best_score = score
            best_points = points

    # OpenCV 局部分割失敗時，使用 bbox 作可編輯矩形候選，
    # 避免 GPT 直接產生巨大錯誤 polygon。
    if not best_points:
        best_points = [
            (float(x), float(y)),
            (float(x + w), float(y)),
            (float(x + w), float(y + h)),
            (float(x), float(y + h)),
        ]

    return best_points


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
        polygon = Polygon(candidate["points"])
        if any(
            _iou(polygon, Polygon(existing["points"])) >= 0.48
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
    """GPT 只辨識空間名稱與 bbox；OpenCV 將 bbox 細化成候選邊界。"""
    if not api_key:
        raise ValueError("尚未設定 OPENAI_API_KEY。")

    options = options or AIRoomDetectionOptions()
    source = image.convert("RGB")

    prompt = f"""
你是建築平面圖空間辨識助手。請只辨識每個實際空間的名稱與大致 bbox，
不要直接輸出 polygon。

bbox 規則：
- 使用完整輸入圖片的 0～1000 座標。
- 每個 bbox 只包住一個空間，盡量貼近該房間內牆。
- 不可產生跨越多個房間的大框。
- 開放式客餐廳若沒有牆體分隔，視為同一空間。
- 忽略尺寸線、文字、家具、床、桌椅、櫃體分格、門片開啟弧、
  樓梯踏階、窗框內線及基地界線。
- 小型衛浴、走道及儲藏室也要檢查。
- 不確定時降低 confidence，不要虛構。

納入面積：
陽台={"是" if options.include_balcony else "否"}；
走道／玄關={"是" if options.include_corridor else "否"}；
衛浴={"是" if options.include_bathroom else "否"}；
樓梯={"是" if options.include_stair else "否"}。

最多輸出 {options.maximum_rooms} 個空間。
"""

    result = _request(
        OpenAI(api_key=api_key),
        model,
        prompt,
        source,
        options.max_retries,
    )

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    image_area = float(source.width * source.height)

    for room in result.get("rooms", []):
        confidence = float(room.get("confidence", 0))
        bbox = _bbox_to_pixels(
            room["bbox"],
            source.width,
            source.height,
        )

        if confidence < options.minimum_confidence:
            rejected.append({
                "room_name": room.get("room_name", "未命名"),
                "confidence": confidence,
                "rejected_reason": "GPT bbox 信心分數過低",
            })
            continue

        points = _local_boundary_from_bbox(source, bbox)
        area_px2 = float(Polygon(points).area) if len(points) >= 3 else 0.0

        reasons: list[str] = []
        if len(points) < 3:
            reasons.append("邊界無效")
        if area_px2 < image_area * options.minimum_polygon_area_ratio:
            reasons.append("面積過小")
        if area_px2 > image_area * 0.28:
            reasons.append("面積異常過大")

        record = {
            "room_id": "",
            "room_name": room.get("room_name") or "未命名空間",
            "room_type": room.get("room_type") or "無法判斷",
            "include_in_area": _include_by_type(room, options),
            "confidence": confidence,
            "points": points,
            "area_px2": area_px2,
            "reason": room.get("reason", ""),
            "source": "gpt_bbox_opencv_boundary",
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
            "is_floor_plan": result.get("assessment", {}).get(
                "is_floor_plan", True
            ),
            "building_region_found": True,
            "quality": result.get("assessment", {}).get("quality", "usable"),
            "note": result.get("assessment", {}).get("note", ""),
        },
        "overall_note": (
            "GPT 只負責空間名稱與 bbox；"
            "OpenCV 依 bbox 嘗試貼合局部封閉邊界。"
        ),
        "model": model,
        "image_width": source.width,
        "image_height": source.height,
    }
