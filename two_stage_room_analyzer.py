from __future__ import annotations

import base64
import io
import json
import os
from copy import deepcopy
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw


DEFAULT_MODEL = os.getenv(
    "OPENAI_VISION_MODEL",
    "gpt-4.1",
)
NORMALIZED_MAX = 1000.0


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0, le=1000)
    y: float = Field(ge=0, le=1000)


class NormalizedBox(BaseModel):
    left: float = Field(ge=0, le=1000)
    top: float = Field(ge=0, le=1000)
    right: float = Field(ge=0, le=1000)
    bottom: float = Field(ge=0, le=1000)


class RoomCandidate(BaseModel):
    id: str
    name: str
    bbox: NormalizedBox
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class Stage1Result(BaseModel):
    rooms: list[RoomCandidate]


class LocalPolygon(BaseModel):
    room_id: str
    room_name: str
    points: list[NormalizedPoint]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class LocalReviewOperation(BaseModel):
    type: Literal[
        "approve",
        "move_point",
        "add_point",
        "delete_point",
    ]
    point_index: int | None = None
    position: NormalizedPoint | None = None
    reason: str = ""


class LocalReviewResult(BaseModel):
    status: Literal[
        "approved",
        "revise",
    ]
    operations: list[LocalReviewOperation]
    summary: str = ""


STAGE1_PROMPT = """
你是一位建築平面圖空間判讀人員。

這是第一階段。你的任務只有：
1. 列出主要室內使用空間。
2. 為每個空間提供大致 bounding box。
3. 不要產生 Polygon。

座標使用 0～1000 標準化座標：
- 左上角 (0,0)
- 右下角 (1000,1000)

請遵守：
1. 只列出實際房間或完整開放式使用空間。
2. 不要把床、桌、櫃體、衛浴設備、樓梯踏階、電梯、門片、陽台、庭院、車道、基地空白視為房間。
3. 開放式客廳、餐廳與廚房若沒有實牆分隔，可視為同一使用空間。
4. bbox 要涵蓋該空間與周邊少量牆線，供第二階段局部裁切使用。
5. 不要求沿牆精準，只需提供合理位置。
6. 房間 ID 依空間排序，例如 room1、room2。
7. 無法確認的候選請降低 confidence。
"""


LOCAL_POLYGON_PROMPT = """
這是第二階段。你現在只處理一個房間。

房間資訊：
- ID：{room_id}
- 名稱：{room_name}

輸入圖片是此房間周邊的局部放大裁切，並加入 0～1000 座標格線。

請只回傳這一個房間的 Polygon。

規則：
1. 所有座標使用局部裁切圖片的 0～1000 標準化座標。
2. Polygon 沿房間牆內側主要轉折。
3. 不得框到相鄰房間、走道、樓梯、家具、櫃體、設備、尺寸線或室外。
4. 門洞視為牆面連續邊界的一部分，不要讓 Polygon 從門洞流出。
5. L 型空間使用 L 型 Polygon。
6. 至少 4 個角點，依順時針或逆時針排列。
7. 不要把局部裁切圖片邊界直接當作房間邊界，除非牆面確實位於該處。
8. 若此候選不是有效房間，回傳低 confidence 與空 points。
"""


LOCAL_REVIEW_PROMPT = """
你正在檢查單一房間 Polygon。

輸入包括：
1. 房間局部裁切與 0～1000 格線。
2. 已畫上紅色 Polygon 與藍色角點編號的疊圖。

請檢查 Polygon 是否：
- 穿過牆。
- 流入相鄰空間。
- 包含家具、樓梯、電梯、櫃體或室外。
- 漏掉房間主要轉折。

你只能回傳：
- approve
- move_point
- add_point
- delete_point

所有修正位置維持局部圖片的 0～1000 座標。
不要重新輸出完整 Polygon。

目前 Polygon：
{polygon_json}
"""


def image_to_data_url(
    image: Image.Image,
) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="PNG",
        optimize=True,
    )
    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def clamp_norm(value: float) -> float:
    return max(
        0.0,
        min(
            NORMALIZED_MAX,
            float(value),
        ),
    )


def norm_to_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        x / 1000 * max(width - 1, 1),
        y / 1000 * max(height - 1, 1),
    )


def pixel_to_norm(
    x: float,
    y: float,
    width: int,
    height: int,
) -> list[float]:
    return [
        clamp_norm(
            x / max(width - 1, 1) * 1000
        ),
        clamp_norm(
            y / max(height - 1, 1) * 1000
        ),
    ]


def add_grid(
    image: Image.Image,
    step: int = 100,
) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    width, height = result.size

    for value in range(0, 1001, step):
        x = round(
            value / 1000 * (width - 1)
        )
        y = round(
            value / 1000 * (height - 1)
        )

        alpha = (
            110
            if value % 500 == 0
            else 45
        )

        draw.line(
            (x, 0, x, height - 1),
            fill=(0, 90, 255, alpha),
            width=(
                2
                if value % 500 == 0
                else 1
            ),
        )

        draw.line(
            (0, y, width - 1, y),
            fill=(0, 90, 255, alpha),
            width=(
                2
                if value % 500 == 0
                else 1
            ),
        )

        draw.rectangle(
            (
                x,
                0,
                min(x + 40, width - 1),
                18,
            ),
            fill=(255, 255, 255, 220),
        )

        draw.text(
            (x + 2, 2),
            str(value),
            fill=(0, 70, 200, 255),
        )

        draw.rectangle(
            (
                0,
                y,
                44,
                min(y + 18, height - 1),
            ),
            fill=(255, 255, 255, 220),
        )

        draw.text(
            (2, y + 2),
            str(value),
            fill=(0, 70, 200, 255),
        )

    return result


def detect_room_candidates(
    image: Image.Image,
    model: str,
) -> list[dict]:
    client = OpenAI()
    gridded = add_grid(image)

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": STAGE1_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            gridded
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=Stage1Result,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "第一階段沒有回傳可解析結果。"
        )

    candidates = []

    for room in response.output_parsed.rooms:
        left = min(
            room.bbox.left,
            room.bbox.right,
        )
        right = max(
            room.bbox.left,
            room.bbox.right,
        )
        top = min(
            room.bbox.top,
            room.bbox.bottom,
        )
        bottom = max(
            room.bbox.top,
            room.bbox.bottom,
        )

        if (
            right - left < 20
            or bottom - top < 20
        ):
            continue

        candidates.append(
            {
                "id": room.id.strip(),
                "name": room.name.strip(),
                "bbox": {
                    "left": clamp_norm(left),
                    "top": clamp_norm(top),
                    "right": clamp_norm(right),
                    "bottom": clamp_norm(bottom),
                },
                "confidence": float(
                    room.confidence
                ),
                "reason": room.reason,
            }
        )

    return candidates


def expand_candidate_box(
    bbox: dict,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    left, top = norm_to_pixel(
        bbox["left"],
        bbox["top"],
        width,
        height,
    )
    right, bottom = norm_to_pixel(
        bbox["right"],
        bbox["bottom"],
        width,
        height,
    )

    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)

    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    x0 = max(0, round(left - pad_x))
    y0 = max(0, round(top - pad_y))
    x1 = min(width, round(right + pad_x))
    y1 = min(height, round(bottom + pad_y))

    return x0, y0, x1, y1


def detect_local_polygon(
    local_grid: Image.Image,
    room_id: str,
    room_name: str,
    model: str,
) -> dict:
    client = OpenAI()

    prompt = LOCAL_POLYGON_PROMPT.format(
        room_id=room_id,
        room_name=room_name,
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            local_grid
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=LocalPolygon,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            f"{room_id} 第二階段沒有回傳可解析結果。"
        )

    parsed = response.output_parsed

    return {
        "id": room_id,
        "name": room_name,
        "points_local_normalized": [
            [
                clamp_norm(point.x),
                clamp_norm(point.y),
            ]
            for point in parsed.points
        ],
        "confidence": float(
            parsed.confidence
        ),
        "notes": parsed.notes,
    }


def draw_local_overlay(
    local_grid: Image.Image,
    polygon: dict,
) -> Image.Image:
    result = local_grid.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    width, height = result.size

    points = [
        norm_to_pixel(
            point[0],
            point[1],
            width,
            height,
        )
        for point in polygon[
            "points_local_normalized"
        ]
    ]

    if len(points) >= 3:
        draw.polygon(
            points,
            fill=(255, 0, 0, 35),
        )
        draw.line(
            points + [points[0]],
            fill=(255, 0, 0, 255),
            width=5,
            joint="curve",
        )

        for index, (x, y) in enumerate(
            points
        ):
            radius = 7

            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                fill=(0, 100, 255, 255),
                outline=(255, 255, 255, 255),
                width=2,
            )

            draw.text(
                (x + 9, y - 10),
                str(index),
                fill=(0, 70, 220, 255),
            )

    return result


def review_local_polygon(
    local_grid: Image.Image,
    polygon: dict,
    model: str,
) -> LocalReviewResult:
    client = OpenAI()
    overlay = draw_local_overlay(
        local_grid,
        polygon,
    )

    prompt = LOCAL_REVIEW_PROMPT.format(
        polygon_json=json.dumps(
            polygon,
            ensure_ascii=False,
            indent=2,
        )
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            local_grid
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            overlay
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=LocalReviewResult,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "局部 Reviewer 沒有回傳可解析結果。"
        )

    return response.output_parsed


def apply_local_review(
    polygon: dict,
    review: LocalReviewResult,
) -> dict:
    result = deepcopy(polygon)
    points = result[
        "points_local_normalized"
    ]

    for operation in review.operations:
        if operation.type == "approve":
            continue

        if operation.type == "move_point":
            if (
                operation.point_index is not None
                and operation.position is not None
                and 0
                <= operation.point_index
                < len(points)
            ):
                points[
                    operation.point_index
                ] = [
                    clamp_norm(
                        operation.position.x
                    ),
                    clamp_norm(
                        operation.position.y
                    ),
                ]

        elif operation.type == "add_point":
            if operation.position is not None:
                index = (
                    len(points)
                    if operation.point_index is None
                    else operation.point_index
                )

                index = max(
                    0,
                    min(len(points), index),
                )

                points.insert(
                    index,
                    [
                        clamp_norm(
                            operation.position.x
                        ),
                        clamp_norm(
                            operation.position.y
                        ),
                    ],
                )

        elif operation.type == "delete_point":
            if (
                operation.point_index is not None
                and len(points) > 3
                and 0
                <= operation.point_index
                < len(points)
            ):
                points.pop(
                    operation.point_index
                )

    return result


def local_to_global_points(
    points_local: list[list[float]],
    crop_box: tuple[int, int, int, int],
) -> list[list[float]]:
    x0, y0, x1, y1 = crop_box
    crop_width = max(x1 - x0, 1)
    crop_height = max(y1 - y0, 1)

    result = []

    for nx, ny in points_local:
        gx = (
            x0
            + nx / 1000
            * max(crop_width - 1, 1)
        )
        gy = (
            y0
            + ny / 1000
            * max(crop_height - 1, 1)
        )

        result.append([gx, gy])

    return result


def polygon_area_pixels(
    points: list[list[float]],
) -> float:
    if len(points) < 3:
        return 0.0

    total = 0.0

    for index, (x1, y1) in enumerate(
        points
    ):
        x2, y2 = points[
            (index + 1) % len(points)
        ]

        total += (
            x1 * y2
            - x2 * y1
        )

    return abs(total) / 2.0


def draw_stage1_overlay(
    image: Image.Image,
    candidates: list[dict],
) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    width, height = result.size

    for candidate in candidates:
        bbox = candidate["bbox"]

        left, top = norm_to_pixel(
            bbox["left"],
            bbox["top"],
            width,
            height,
        )

        right, bottom = norm_to_pixel(
            bbox["right"],
            bbox["bottom"],
            width,
            height,
        )

        draw.rectangle(
            (left, top, right, bottom),
            outline=(255, 140, 0, 255),
            width=4,
        )

        draw.text(
            (left + 5, top + 5),
            (
                f'{candidate["id"]} '
                f'{candidate["name"]}'
            ),
            fill=(255, 90, 0, 255),
        )

    return result


def draw_final_overlay(
    image: Image.Image,
    rooms: list[dict],
) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")

    for room in rooms:
        points = [
            tuple(point)
            for point in room["points"]
        ]

        if len(points) < 3:
            continue

        draw.polygon(
            points,
            fill=(255, 0, 0, 30),
        )

        draw.line(
            points + [points[0]],
            fill=(255, 0, 0, 255),
            width=5,
            joint="curve",
        )

        x, y = points[0]

        draw.text(
            (x + 8, y + 8),
            f'{room["id"]} {room["name"]}',
            fill=(255, 0, 0, 255),
        )

    return result


def run_two_stage_analysis(
    image: Image.Image,
    model: str = DEFAULT_MODEL,
    crop_padding_ratio: float = 0.18,
    local_review_rounds: int = 1,
) -> dict:
    image = image.convert("RGB")
    width, height = image.size

    candidates = detect_room_candidates(
        image,
        model=model,
    )

    rooms = []
    local_crops = []
    logs = []

    for candidate in candidates:
        crop_box = expand_candidate_box(
            candidate["bbox"],
            width,
            height,
            crop_padding_ratio,
        )

        local_image = image.crop(crop_box)
        local_grid = add_grid(local_image)

        polygon = detect_local_polygon(
            local_grid,
            room_id=candidate["id"],
            room_name=candidate["name"],
            model=model,
        )

        room_log = {
            "room_id": candidate["id"],
            "room_name": candidate["name"],
            "crop_box": list(crop_box),
            "initial_polygon": deepcopy(
                polygon
            ),
            "reviews": [],
        }

        for round_index in range(
            local_review_rounds
        ):
            if len(
                polygon[
                    "points_local_normalized"
                ]
            ) < 3:
                break

            review = review_local_polygon(
                local_grid,
                polygon,
                model=model,
            )

            room_log["reviews"].append(
                {
                    "round": round_index + 1,
                    "review": review.model_dump(),
                }
            )

            if review.status == "approved":
                break

            updated = apply_local_review(
                polygon,
                review,
            )

            if updated == polygon:
                break

            polygon = updated

        global_points = local_to_global_points(
            polygon[
                "points_local_normalized"
            ],
            crop_box,
        )

        if len(global_points) >= 3:
            rooms.append(
                {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "confidence": polygon[
                        "confidence"
                    ],
                    "points": global_points,
                    "points_local_normalized": polygon[
                        "points_local_normalized"
                    ],
                    "crop_box": list(crop_box),
                    "area_pixels": (
                        polygon_area_pixels(
                            global_points
                        )
                    ),
                    "notes": polygon["notes"],
                }
            )

        local_crops.append(
            {
                "room_id": candidate["id"],
                "room_name": candidate["name"],
                "crop_box": list(crop_box),
                "image": draw_local_overlay(
                    local_grid,
                    polygon,
                ),
            }
        )

        logs.append(room_log)

    return {
        "stage1_candidates": candidates,
        "rooms": rooms,
        "logs": logs,
        "image_size": [width, height],
        "stage1_overlay": draw_stage1_overlay(
            image,
            candidates,
        ),
        "final_overlay": draw_final_overlay(
            image,
            rooms,
        ),
        "local_crops": local_crops,
    }
