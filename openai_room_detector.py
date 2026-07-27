from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image
from shapely.geometry import Polygon


ROOM_DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_assessment": {
            "type": "object",
            "properties": {
                "is_floor_plan": {"type": "boolean"},
                "building_region_found": {"type": "boolean"},
                "quality": {
                    "type": "string",
                    "enum": ["good", "usable", "poor"],
                },
                "note": {"type": "string"},
            },
            "required": [
                "is_floor_plan",
                "building_region_found",
                "quality",
                "note",
            ],
            "additionalProperties": False,
        },
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "room_id": {"type": "string"},
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
                                "x": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                                "y": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                            },
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "room_id",
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
    "required": [
        "image_assessment",
        "rooms",
        "excluded_spaces",
        "overall_note",
    ],
    "additionalProperties": False,
}


@dataclass
class AIRoomDetectionOptions:
    include_balcony: bool = True
    include_corridor: bool = True
    include_stair: bool = False
    include_bathroom: bool = True
    minimum_confidence: float = 0.35
    minimum_polygon_area_ratio: float = 0.0008
    maximum_rooms: int = 30


def _to_data_url(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _create_analysis_image(image: Image.Image) -> Image.Image:
    """建立第二張高對比圖，協助模型區分牆線、家具及文字。

    座標仍以原始傳入圖片為準，第二張圖片只作視覺輔助。
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 保留灰階細節並提高局部對比，避免直接二值化造成牆線與家具全部黏在一起。
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 將深色線條稍微加深，背景保持白色。
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result)


def _deduplicate_points(
    points: list[tuple[float, float]],
    tolerance: float = 1.5,
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for point in points:
        if not output:
            output.append(point)
            continue
        if (
            abs(point[0] - output[-1][0]) > tolerance
            or abs(point[1] - output[-1][1]) > tolerance
        ):
            output.append(point)

    if (
        len(output) > 2
        and abs(output[0][0] - output[-1][0]) <= tolerance
        and abs(output[0][1] - output[-1][1]) <= tolerance
    ):
        output.pop()
    return output


def _repair_polygon(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    points = [
        (
            min(max(float(x), 0.0), float(width - 1)),
            min(max(float(y), 0.0), float(height - 1)),
        )
        for x, y in points
    ]
    points = _deduplicate_points(points)
    if len(points) < 3:
        return []

    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return []

    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)

    # 避免模型回傳過度密集的鋸齒座標。
    polygon = polygon.simplify(
        max(width, height) * 0.0015,
        preserve_topology=True,
    )
    if polygon.is_empty:
        return []

    repaired = [
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]
    return _deduplicate_points(repaired)


def _normalized_to_pixel_polygon(
    normalized_points: list[dict[str, int]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    points = [
        (
            float(point["x"]) / 1000.0 * width,
            float(point["y"]) / 1000.0 * height,
        )
        for point in normalized_points
    ]
    return _repair_polygon(points, width, height)


def _should_include_room(
    room: dict[str, Any],
    options: AIRoomDetectionOptions,
) -> bool:
    room_type = str(room.get("room_type", "")).lower()
    room_name = str(room.get("room_name", "")).lower()

    text = f"{room_type} {room_name}"

    if any(term in text for term in ["stair", "樓梯", "梯間"]):
        return options.include_stair
    if any(term in text for term in ["balcony", "陽台", "露台"]):
        return options.include_balcony
    if any(term in text for term in ["corridor", "走道", "走廊", "玄關"]):
        return options.include_corridor
    if any(term in text for term in ["bathroom", "toilet", "衛浴", "浴室", "廁所"]):
        return options.include_bathroom

    return bool(room.get("include_in_area", True))


def detect_rooms_with_openai(
    api_key: str,
    image: Image.Image,
    model: str = "gpt-4.1",
    options: AIRoomDetectionOptions | None = None,
) -> dict[str, Any]:
    """直接使用 OpenAI Vision 辨識房間並回傳可編輯多邊形。

    回傳 polygon 座標已轉為傳入圖片的實際像素座標。
    """
    if not api_key:
        raise ValueError("尚未設定 OPENAI_API_KEY。")

    options = options or AIRoomDetectionOptions()
    source = image.convert("RGB")
    analysis_image = _create_analysis_image(source)

    prompt = f"""
你是建築平面圖空間辨識助手。請直接從平面圖辨識由牆體圍合的室內空間，
並為每一個空間回傳可供人工修改的多邊形。

你會收到兩張內容相同的圖：
- 第一張：原始平面圖，所有 polygon 座標必須以第一張圖為準。
- 第二張：高對比輔助圖，只用來幫助辨識線條。

座標規則：
1. polygon 使用 0 到 1000 的正規化整數座標。
2. 左上角為 (0,0)，右下角為 (1000,1000)。
3. polygon 點位沿房間「內牆完成面」順時針或逆時針排列。
4. 不要使用房間外接矩形；L 型、凹型空間應依實際形狀提供轉折點。
5. 每一個 polygon 最少 4 點、最多 24 點。
6. 點位不必達到 CAD 精度，但必須盡量貼近內牆，而不是尺寸線、家具或文字。

辨識規則：
- 牆體是空間邊界。
- 忽略尺寸線、標註數字、家具、櫃體內部分格、床、桌椅、樓梯踏階線、
  門片開啟弧線、窗戶內部線及基地界線。
- 開放式客廳與餐廳若沒有牆體分隔，視為同一空間。
- 門洞不代表兩個空間相連；應沿牆面推估門洞處的房間邊界。
- 樓梯本體不是一般冷房空間，除非它位於封閉梯廳。
- 同一房間只能輸出一次，不要重疊輸出。
- 不能確定時降低 confidence，不要虛構房間。

納入設定：
- 陽台：{"納入" if options.include_balcony else "不納入"}
- 走道及玄關：{"納入" if options.include_corridor else "不納入"}
- 樓梯：{"納入" if options.include_stair else "不納入"}
- 衛浴：{"納入" if options.include_bathroom else "不納入"}

最多輸出 {options.maximum_rooms} 個空間。
這是候選框產生工作，不得宣稱為正式測量或 CAD 精度。
"""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _to_data_url(source),
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": _to_data_url(analysis_image),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "floorplan_room_detection",
                "strict": True,
                "schema": ROOM_DETECTION_SCHEMA,
            }
        },
    )

    raw = json.loads(response.output_text)
    width, height = source.size
    image_area = float(width * height)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, room in enumerate(raw.get("rooms", []), start=1):
        confidence = float(room.get("confidence", 0.0))
        polygon = _normalized_to_pixel_polygon(
            room.get("polygon", []),
            width,
            height,
        )

        area_px2 = 0.0
        if len(polygon) >= 3:
            area_px2 = float(Polygon(polygon).area)

        reasons: list[str] = []
        if confidence < options.minimum_confidence:
            reasons.append("信心分數低於設定值")
        if len(polygon) < 3:
            reasons.append("多邊形座標無效")
        if area_px2 < image_area * options.minimum_polygon_area_ratio:
            reasons.append("候選空間面積過小")

        record = {
            "room_id": room.get("room_id") or f"R{index:02d}",
            "room_name": room.get("room_name") or f"空間{index}",
            "room_type": room.get("room_type") or "無法判斷",
            "include_in_area": _should_include_room(room, options),
            "confidence": confidence,
            "points": polygon,
            "area_px2": area_px2,
            "reason": room.get("reason", ""),
            "raw_polygon": room.get("polygon", []),
        }

        if reasons:
            record["rejected_reason"] = "、".join(reasons)
            rejected.append(record)
        else:
            accepted.append(record)

    # 依照圖面由上到下、由左到右重新編號，讓畫布與表格順序一致。
    accepted.sort(
        key=lambda room: (
            min((y for _, y in room["points"]), default=0),
            min((x for x, _ in room["points"]), default=0),
        )
    )
    for index, room in enumerate(accepted, start=1):
        room["room_id"] = f"R{index:02d}"

    return {
        "rooms": accepted,
        "rejected_rooms": rejected,
        "excluded_spaces": raw.get("excluded_spaces", []),
        "image_assessment": raw.get("image_assessment", {}),
        "overall_note": raw.get("overall_note", ""),
        "model": model,
        "image_width": width,
        "image_height": height,
    }
