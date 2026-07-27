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


TILE_ROOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tile_assessment": {
            "type": "object",
            "properties": {
                "contains_floor_plan": {"type": "boolean"},
                "quality": {
                    "type": "string",
                    "enum": ["good", "usable", "poor"],
                },
                "note": {"type": "string"},
            },
            "required": ["contains_floor_plan", "quality", "note"],
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
    },
    "required": ["tile_assessment", "rooms"],
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

    # 混合辨識參數
    max_tiles: int = 6
    tile_overlap_ratio: float = 0.18
    min_tile_ink_ratio: float = 0.008
    max_retries: int = 2


@dataclass
class TileProposal:
    tile_id: str
    core_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    ink_ratio: float


def _to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _binary_ink(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    _, global_mask = cv2.threshold(
        gray,
        220,
        255,
        cv2.THRESH_BINARY_INV,
    )
    ink = cv2.bitwise_and(
        adaptive,
        cv2.dilate(global_mask, np.ones((2, 2), np.uint8)),
    )
    return cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )


def _locate_building_bbox(
    image: Image.Image,
    ink: np.ndarray,
) -> tuple[int, int, int, int]:
    """OpenCV 粗定位建築主體，不在此階段判斷房間。"""
    height, width = ink.shape
    image_area = float(width * height)

    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
    )
    structural = cv2.bitwise_or(horizontal, vertical)
    structural = cv2.dilate(
        structural,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
        iterations=1,
    )
    structural = cv2.morphologyEx(
        structural,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        structural,
        8,
    )

    center_x = width / 2
    center_y = height / 2
    best_box = None
    best_score = -1.0

    for label in range(1, count):
        x, y, box_w, box_h, area = stats[label]
        box_area = float(box_w * box_h)
        ratio = box_area / max(image_area, 1.0)

        if ratio < 0.025 or ratio > 0.92:
            continue
        if min(box_w, box_h) < 100:
            continue

        source_ink = np.count_nonzero(
            ink[y:y + box_h, x:x + box_w]
        )
        density = source_ink / max(box_area, 1.0)

        cx, cy = centroids[label]
        distance = math.hypot(
            (cx - center_x) / max(width, 1),
            (cy - center_y) / max(height, 1),
        )
        center_weight = max(0.25, 1.0 - distance)

        aspect = max(
            box_w / max(box_h, 1),
            box_h / max(box_w, 1),
        )
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
    padding = max(12, round(min(box_w, box_h) * 0.035))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + box_w + padding)
    y1 = min(height, y + box_h + padding)
    return x0, y0, x1 - x0, y1 - y0


def _choose_grid(
    width: int,
    height: int,
    max_tiles: int,
) -> tuple[int, int]:
    max_tiles = max(1, min(int(max_tiles), 9))

    if max_tiles == 1:
        return 1, 1

    candidates: list[tuple[float, int, int]] = []
    image_aspect = width / max(height, 1)

    for rows in range(1, 4):
        for cols in range(1, 4):
            tiles = rows * cols
            if tiles > max_tiles:
                continue
            tile_aspect = (width / cols) / max(height / rows, 1)
            aspect_penalty = abs(math.log(max(tile_aspect, 0.01)))
            unused_penalty = (max_tiles - tiles) * 0.08
            orientation_penalty = abs(
                math.log(max(image_aspect, 0.01))
                - math.log(max(cols / rows, 0.01))
            ) * 0.15
            score = aspect_penalty + unused_penalty + orientation_penalty
            candidates.append((score, rows, cols))

    _, rows, cols = min(candidates)
    return rows, cols


def propose_tiles(
    image: Image.Image,
    options: AIRoomDetectionOptions | None = None,
) -> tuple[list[TileProposal], dict[str, Any]]:
    """由 OpenCV 定位建築範圍，再切成帶上下文的重疊區塊。"""
    options = options or AIRoomDetectionOptions()
    ink = _binary_ink(image)
    bx, by, bw, bh = _locate_building_bbox(image, ink)
    rows, cols = _choose_grid(bw, bh, options.max_tiles)

    proposals: list[TileProposal] = []
    tile_number = 0

    for row in range(rows):
        core_y0 = by + round(bh * row / rows)
        core_y1 = by + round(bh * (row + 1) / rows)

        for col in range(cols):
            core_x0 = bx + round(bw * col / cols)
            core_x1 = bx + round(bw * (col + 1) / cols)

            core_w = core_x1 - core_x0
            core_h = core_y1 - core_y0
            margin_x = round(core_w * options.tile_overlap_ratio)
            margin_y = round(core_h * options.tile_overlap_ratio)

            crop_x0 = max(0, core_x0 - margin_x)
            crop_y0 = max(0, core_y0 - margin_y)
            crop_x1 = min(image.width, core_x1 + margin_x)
            crop_y1 = min(image.height, core_y1 + margin_y)

            crop_ink = ink[crop_y0:crop_y1, crop_x0:crop_x1]
            ink_ratio = (
                float(np.count_nonzero(crop_ink))
                / max(float(crop_ink.size), 1.0)
            )

            if ink_ratio < options.min_tile_ink_ratio:
                continue

            tile_number += 1
            proposals.append(
                TileProposal(
                    tile_id=f"T{tile_number:02d}",
                    core_box=(
                        core_x0,
                        core_y0,
                        core_x1 - core_x0,
                        core_y1 - core_y0,
                    ),
                    crop_box=(
                        crop_x0,
                        crop_y0,
                        crop_x1 - crop_x0,
                        crop_y1 - crop_y0,
                    ),
                    ink_ratio=ink_ratio,
                )
            )

    debug = {
        "building_box": (bx, by, bw, bh),
        "rows": rows,
        "cols": cols,
        "ink": ink,
    }
    return proposals, debug


def draw_tile_proposals(
    image: Image.Image,
    proposals: list[TileProposal],
    building_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    if building_box:
        x, y, w, h = building_box
        draw.rectangle(
            (x, y, x + w, y + h),
            outline="#EF4444",
            width=4,
        )

    for proposal in proposals:
        cx, cy, cw, ch = proposal.core_box
        tx, ty, tw, th = proposal.crop_box

        draw.rectangle(
            (tx, ty, tx + tw, ty + th),
            outline="#94A3B8",
            width=2,
        )
        draw.rectangle(
            (cx, cy, cx + cw, cy + ch),
            outline="#2563EB",
            width=3,
        )
        draw.rectangle(
            (cx + 4, cy + 4, cx + 52, cy + 26),
            fill="white",
            outline="#2563EB",
            width=1,
        )
        draw.text(
            (cx + 9, cy + 8),
            proposal.tile_id,
            fill="#2563EB",
        )

    return output


def _analysis_crop(crop: Image.Image) -> Image.Image:
    rgb = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    return Image.fromarray(
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    )


def _repair_polygon(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return []

    points = [
        (
            min(max(float(x), 0.0), image_width - 1.0),
            min(max(float(y), 0.0), image_height - 1.0),
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
        max(image_width, image_height) * 0.0012,
        preserve_topology=True,
    )
    if polygon.is_empty:
        return []

    return [
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]


def _crop_polygon_to_global(
    normalized: list[dict[str, int]],
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    crop_x, crop_y, crop_w, crop_h = crop_box
    points = [
        (
            crop_x + float(point["x"]) / 1000.0 * crop_w,
            crop_y + float(point["y"]) / 1000.0 * crop_h,
        )
        for point in normalized
    ]
    return _repair_polygon(
        points,
        image_width,
        image_height,
    )


def _centroid_in_core(
    points: list[tuple[float, float]],
    core_box: tuple[int, int, int, int],
    tolerance: float = 3.0,
) -> bool:
    if len(points) < 3:
        return False

    centroid = Polygon(points).centroid
    x, y, w, h = core_box
    return (
        x - tolerance <= centroid.x <= x + w + tolerance
        and y - tolerance <= centroid.y <= y + h + tolerance
    )


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


def _iou(
    first: Polygon,
    second: Polygon,
) -> float:
    union = first.union(second).area
    if union <= 0:
        return 0.0
    return float(first.intersection(second).area / union)


def _deduplicate_rooms(
    rooms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rooms = sorted(
        rooms,
        key=lambda room: (
            float(room.get("confidence", 0)),
            float(room.get("area_px2", 0)),
        ),
        reverse=True,
    )

    accepted: list[dict[str, Any]] = []

    for candidate in rooms:
        candidate_polygon = Polygon(candidate["points"])
        duplicate_index = None

        for index, existing in enumerate(accepted):
            existing_polygon = Polygon(existing["points"])
            overlap = _iou(candidate_polygon, existing_polygon)

            centroid_distance = candidate_polygon.centroid.distance(
                existing_polygon.centroid
            )
            scale = math.sqrt(
                max(
                    min(candidate_polygon.area, existing_polygon.area),
                    1.0,
                )
            )
            same_center = centroid_distance / scale < 0.22
            similar_area = (
                min(candidate_polygon.area, existing_polygon.area)
                / max(candidate_polygon.area, existing_polygon.area)
                > 0.55
            )

            if overlap >= 0.38 or (same_center and similar_area):
                duplicate_index = index
                break

        if duplicate_index is None:
            accepted.append(candidate)
            continue

        existing = accepted[duplicate_index]
        if candidate["confidence"] > existing["confidence"]:
            accepted[duplicate_index] = candidate

    accepted.sort(
        key=lambda room: (
            min(y for _, y in room["points"]),
            min(x for x, _ in room["points"]),
        )
    )

    for index, room in enumerate(accepted, start=1):
        room["room_id"] = f"R{index:02d}"

    return accepted


def _call_tile(
    client: OpenAI,
    model: str,
    crop: Image.Image,
    proposal: TileProposal,
    options: AIRoomDetectionOptions,
) -> dict[str, Any]:
    crop_x, crop_y, crop_w, crop_h = proposal.crop_box
    core_x, core_y, core_w, core_h = proposal.core_box

    local_core = {
        "left": round((core_x - crop_x) / crop_w * 1000),
        "top": round((core_y - crop_y) / crop_h * 1000),
        "right": round((core_x + core_w - crop_x) / crop_w * 1000),
        "bottom": round((core_y + core_h - crop_y) / crop_h * 1000),
    }

    prompt = f"""
你是建築平面圖局部區塊的空間辨識助手。

這不是整張圖，而是 OpenCV 從建築主體切出的 {proposal.tile_id} 局部圖。
圖中包含：
- 藍色核心責任區（以文字座標表示，不會真的畫在圖上）
- 核心區四周的重疊上下文，用來看清完整牆體

核心責任區在本裁切圖的 0～1000 座標為：
left={local_core["left"]}, top={local_core["top"]},
right={local_core["right"]}, bottom={local_core["bottom"]}

只回傳「房間中心點位於核心責任區內」的空間。
即使房間跨越核心區邊界，也要利用周圍上下文盡量畫出完整房間；
但房間中心不在核心區內時不要回傳，避免不同區塊重複。

polygon 規則：
1. 使用本裁切圖的 0～1000 正規化座標。
2. 左上角為 (0,0)，右下角為 (1000,1000)。
3. 沿房間內牆完成面排列。
4. 不可只畫外接矩形；L 型或凹型空間要保留轉折。
5. 最少 4 點、最多 24 點。
6. 門洞處依牆面方向補成空間邊界。

辨識規則：
- 忽略尺寸線、標註數字、家具、床、桌椅、櫃體內部分格。
- 忽略門片開啟弧、窗戶內部線、樓梯踏階、基地界線。
- 開放式客餐廳若沒有牆體分隔，視為同一空間。
- 不確定時降低 confidence，不要虛構。
- 不得宣稱為 CAD 或正式測量精度。

納入面積設定：
陽台={"是" if options.include_balcony else "否"}；
走道／玄關={"是" if options.include_corridor else "否"}；
衛浴={"是" if options.include_bathroom else "否"}；
樓梯={"是" if options.include_stair else "否"}。
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _to_data_url(crop),
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": _to_data_url(_analysis_crop(crop)),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "floorplan_tile_rooms",
                "strict": True,
                "schema": TILE_ROOM_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def detect_rooms_with_openai(
    api_key: str,
    image: Image.Image,
    model: str = "gpt-4.1",
    options: AIRoomDetectionOptions | None = None,
) -> dict[str, Any]:
    """OpenCV 粗定位＋GPT 局部精辨識。

    OpenCV 只負責：
    1. 找建築主體範圍。
    2. 切出有重疊上下文的區塊。
    3. 將 GPT 的局部座標映射回完整圖面。
    4. 排除低信心、過小及重複候選框。

    GPT 負責：
    1. 分辨牆體、尺寸線、家具、門弧與樓梯踏階。
    2. 判斷空間用途。
    3. 產生局部房間多邊形。
    """
    if not api_key:
        raise ValueError("尚未設定 OPENAI_API_KEY。")

    options = options or AIRoomDetectionOptions()
    source = image.convert("RGB")
    proposals, debug = propose_tiles(source, options)

    if not proposals:
        raise RuntimeError("OpenCV 未找到可送往 GPT 的建築區塊。")

    client = OpenAI(api_key=api_key)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    tile_results: list[dict[str, Any]] = []
    full_area = float(source.width * source.height)

    for tile_index, proposal in enumerate(proposals, start=1):
        x, y, w, h = proposal.crop_box
        crop = source.crop((x, y, x + w, y + h))

        last_error: Exception | None = None
        tile_response = None

        for attempt in range(options.max_retries + 1):
            try:
                tile_response = _call_tile(
                    client,
                    model,
                    crop,
                    proposal,
                    options,
                )
                break
            except Exception as error:
                last_error = error
                if attempt >= options.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))

        if tile_response is None:
            tile_results.append(
                {
                    "tile_id": proposal.tile_id,
                    "status": "error",
                    "error": str(last_error),
                    "rooms": 0,
                }
            )
            continue

        tile_room_count = 0
        for room in tile_response.get("rooms", []):
            points = _crop_polygon_to_global(
                room.get("polygon", []),
                proposal.crop_box,
                source.width,
                source.height,
            )

            reasons: list[str] = []
            confidence = float(room.get("confidence", 0.0))
            area_px2 = (
                float(Polygon(points).area)
                if len(points) >= 3
                else 0.0
            )

            if confidence < options.minimum_confidence:
                reasons.append("低於最低信心分數")
            if len(points) < 3:
                reasons.append("多邊形無效")
            if area_px2 < full_area * options.minimum_polygon_area_ratio:
                reasons.append("面積過小")
            if points and not _centroid_in_core(
                points,
                proposal.core_box,
            ):
                reasons.append("中心點不在本區塊責任區")

            record = {
                "room_id": "",
                "room_name": room.get("room_name") or "未命名空間",
                "room_type": room.get("room_type") or "無法判斷",
                "include_in_area": _include_by_type(room, options),
                "confidence": confidence,
                "points": points,
                "area_px2": area_px2,
                "reason": room.get("reason", ""),
                "source_tile": proposal.tile_id,
            }

            if reasons:
                record["rejected_reason"] = "、".join(reasons)
                rejected.append(record)
            else:
                candidates.append(record)
                tile_room_count += 1

        tile_results.append(
            {
                "tile_id": proposal.tile_id,
                "status": "ok",
                "quality": tile_response.get(
                    "tile_assessment",
                    {},
                ).get("quality", ""),
                "note": tile_response.get(
                    "tile_assessment",
                    {},
                ).get("note", ""),
                "rooms": tile_room_count,
            }
        )

    rooms = _deduplicate_rooms(candidates)
    if len(rooms) > options.maximum_rooms:
        rooms = sorted(
            rooms,
            key=lambda room: room["confidence"],
            reverse=True,
        )[:options.maximum_rooms]
        rooms.sort(
            key=lambda room: (
                min(y for _, y in room["points"]),
                min(x for x, _ in room["points"]),
            )
        )
        for index, room in enumerate(rooms, start=1):
            room["room_id"] = f"R{index:02d}"

    successful_tiles = sum(
        1 for item in tile_results if item["status"] == "ok"
    )
    failed_tiles = len(tile_results) - successful_tiles

    return {
        "rooms": rooms,
        "rejected_rooms": rejected,
        "excluded_spaces": [],
        "image_assessment": {
            "is_floor_plan": True,
            "building_region_found": True,
            "quality": (
                "usable"
                if successful_tiles
                else "poor"
            ),
            "note": (
                f"OpenCV 將建築主體切成 {len(proposals)} 個重疊區塊；"
                f"GPT 成功分析 {successful_tiles} 個，失敗 {failed_tiles} 個。"
            ),
        },
        "overall_note": (
            "本結果採 OpenCV 粗定位與 GPT 局部辨識。"
            "多邊形仍是候選框，請在畫布上確認內牆邊界後再計算面積。"
        ),
        "model": model,
        "image_width": source.width,
        "image_height": source.height,
        "tile_results": tile_results,
        "tile_proposals": [
            {
                "tile_id": proposal.tile_id,
                "core_box": proposal.core_box,
                "crop_box": proposal.crop_box,
                "ink_ratio": proposal.ink_ratio,
            }
            for proposal in proposals
        ],
        "building_box": debug["building_box"],
    }
