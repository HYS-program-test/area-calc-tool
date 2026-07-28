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
    overall_status: Literal[
        "approved",
        "revise",
    ]
    operations: list[ReviewOperation]
    summary: str = ""


INITIAL_PROMPT = """
你是一位建築平面圖空間判讀人員。

輸入圖片已經是建築主體，並覆蓋 0～1000 的標準化座標格線。

座標規則：
1. 所有 x、y 使用 0～1000 標準化座標。
2. 左上角是 (0,0)，右下角是 (1000,1000)。
3. 不使用圖片實際像素座標。
4. 參照格線與數字標籤定位。

空間規則：
1. 框完整房間或完整開放式使用空間。
2. 不框家具、櫃體、樓梯踏階、電梯設備、門片、窗戶、尺寸線、文字、陽台、庭院、車道或建築外部。
3. 開放式客廳、餐廳、廚房若沒有完整隔牆，視為同一空間。
4. 有完整隔牆的臥室、衛浴、儲藏室分別框選。
5. Polygon 沿牆內側主要轉折，不得跨越實牆。
6. L 型空間使用 L 型 Polygon。
7. 每個空間至少 4 個角點，依順時針或逆時針排列。
8. 不要把圖片邊界誤認為牆。
9. 不確定時降低 confidence。
"""


REVIEW_PROMPT = """
你正在執行視覺修正，不是重新產生全部房間。

你會看到：
1. 建築主體與 0～1000 座標格線。
2. 紅色 Polygon 與藍色角點編號疊圖。

請逐一檢查：
- 是否框到家具、設備、樓梯、電梯、尺寸線或室外。
- 是否穿過牆。
- 是否漏掉主要轉折。
- 是否應刪除。
- 名稱是否合理。

只能回傳：
approve、move_point、add_point、delete_point、delete_room、rename_room。

規則：
1. 不重新輸出完整 Polygon。
2. 每輪只修正最明顯且必要的錯誤。
3. 所有位置維持 0～1000 標準化座標。
4. 全部合理時 overall_status=approved。
5. 尚需修正時 overall_status=revise。

目前房間 JSON：
{rooms_json}

目前輪次：{round_index}/{max_rounds}
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


def clamp_normalized(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            NORMALIZED_MAX,
            float(value),
        ),
    )


def normalized_to_pixel(
    point: list[float],
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        point[0]
        / NORMALIZED_MAX
        * max(width - 1, 1),
        point[1]
        / NORMALIZED_MAX
        * max(height - 1, 1),
    )


def add_normalized_grid(
    image: Image.Image,
    step: int = 100,
) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(
        result,
        "RGBA",
    )
    width, height = result.size

    for value in range(
        0,
        1001,
        step,
    ):
        x = round(
            value
            / 1000
            * (width - 1)
        )
        y = round(
            value
            / 1000
            * (height - 1)
        )

        alpha = (
            110
            if value % 500 == 0
            else 45
        )
        line_width = (
            2
            if value % 500 == 0
            else 1
        )

        draw.line(
            (x, 0, x, height - 1),
            fill=(0, 90, 255, alpha),
            width=line_width,
        )
        draw.line(
            (0, y, width - 1, y),
            fill=(0, 90, 255, alpha),
            width=line_width,
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


def clean_rooms(
    rooms: list[RoomPolygon],
) -> list[dict]:
    result = []

    for room in rooms:
        points = [
            [
                clamp_normalized(point.x),
                clamp_normalized(point.y),
            ]
            for point in room.points
        ]

        if len(points) < 3:
            continue

        result.append(
            {
                "id": room.id.strip(),
                "name": room.name.strip(),
                "points": points,
                "confidence": float(
                    room.confidence
                ),
            }
        )

    return result


def detect_initial_rooms(
    gridded_image: Image.Image,
    model: str,
) -> list[dict]:
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": INITIAL_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            gridded_image
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=InitialDetection,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "OpenAI 未回傳可解析的初始 Polygon。"
        )

    return clean_rooms(
        response.output_parsed.rooms
    )


def draw_review_overlay(
    gridded_image: Image.Image,
    rooms: list[dict],
) -> Image.Image:
    overlay = (
        gridded_image
        .copy()
        .convert("RGB")
    )

    draw = ImageDraw.Draw(
        overlay,
        "RGBA",
    )

    width, height = overlay.size

    for room in rooms:
        points = [
            normalized_to_pixel(
                point,
                width,
                height,
            )
            for point in room["points"]
        ]

        if len(points) < 3:
            continue

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
                f'{room["id"]}:{index}',
                fill=(0, 70, 220, 255),
            )

        x, y = points[0]

        draw.text(
            (x + 8, y + 17),
            f'{room["id"]} {room["name"]}',
            fill=(255, 0, 0, 255),
        )

    return overlay


def request_review(
    gridded_image: Image.Image,
    overlay_image: Image.Image,
    rooms: list[dict],
    round_index: int,
    max_rounds: int,
    model: str,
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
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            gridded_image
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            overlay_image
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=ReviewResult,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "OpenAI 未回傳可解析的修正指令。"
        )

    return response.output_parsed


def apply_operations(
    rooms: list[dict],
    review: ReviewResult,
) -> list[dict]:
    result = deepcopy(rooms)

    def find_room(
        room_id: str,
    ) -> dict | None:
        return next(
            (
                room
                for room in result
                if room["id"] == room_id
            ),
            None,
        )

    for operation in review.operations:
        room = find_room(
            operation.room_id
        )

        if operation.type == "delete_room":
            result = [
                candidate
                for candidate in result
                if candidate["id"]
                != operation.room_id
            ]
            continue

        if (
            room is None
            or operation.type == "approve"
        ):
            continue

        if operation.type == "rename_room":
            if operation.new_name:
                room["name"] = (
                    operation
                    .new_name
                    .strip()
                )
            continue

        points = room["points"]

        if operation.type == "move_point":
            if (
                operation.point_index
                is not None
                and operation.position
                is not None
                and 0
                <= operation.point_index
                < len(points)
            ):
                points[
                    operation.point_index
                ] = [
                    clamp_normalized(
                        operation.position.x
                    ),
                    clamp_normalized(
                        operation.position.y
                    ),
                ]

        elif operation.type == "add_point":
            if operation.position is not None:
                index = (
                    len(points)
                    if operation.point_index
                    is None
                    else operation.point_index
                )

                index = max(
                    0,
                    min(
                        len(points),
                        index,
                    ),
                )

                points.insert(
                    index,
                    [
                        clamp_normalized(
                            operation.position.x
                        ),
                        clamp_normalized(
                            operation.position.y
                        ),
                    ],
                )

        elif operation.type == "delete_point":
            if (
                operation.point_index
                is not None
                and len(points) > 3
                and 0
                <= operation.point_index
                < len(points)
            ):
                points.pop(
                    operation.point_index
                )

    return result


def polygon_area_pixels(
    points: list[list[float]],
    width: int,
    height: int,
) -> float:
    pixel_points = [
        normalized_to_pixel(
            point,
            width,
            height,
        )
        for point in points
    ]

    total = 0.0

    for index, (x1, y1) in enumerate(
        pixel_points
    ):
        x2, y2 = pixel_points[
            (index + 1)
            % len(pixel_points)
        ]
        total += (
            x1 * y2
            - x2 * y1
        )

    return abs(total) / 2.0


def run_visual_review_loop(
    image: Image.Image,
    model: str = DEFAULT_MODEL,
    max_rounds: int = 3,
) -> dict:
    """
    app.py 使用位置參數呼叫：
    run_visual_review_loop(building_image, model, max_rounds)

    因此不會再發生 original_image 關鍵字不一致。
    """
    image = image.convert("RGB")
    width, height = image.size

    gridded_image = add_normalized_grid(
        image
    )

    rooms = detect_initial_rooms(
        gridded_image,
        model=model,
    )

    history = []

    for round_index in range(
        1,
        max_rounds + 1,
    ):
        overlay = draw_review_overlay(
            gridded_image,
            rooms,
        )

        review = request_review(
            gridded_image=gridded_image,
            overlay_image=overlay,
            rooms=rooms,
            round_index=round_index,
            max_rounds=max_rounds,
            model=model,
        )

        history.append(
            {
                "round": round_index,
                "rooms_before": deepcopy(
                    rooms
                ),
                "review": (
                    review.model_dump()
                ),
            }
        )

        if (
            review.overall_status
            == "approved"
        ):
            break

        updated = apply_operations(
            rooms,
            review,
        )

        if updated == rooms:
            break

        rooms = updated

    for room in rooms:
        room["area_pixels"] = (
            polygon_area_pixels(
                room["points"],
                width,
                height,
            )
        )

        room["points_pixels"] = [
            list(
                normalized_to_pixel(
                    point,
                    width,
                    height,
                )
            )
            for point in room["points"]
        ]

    return {
        "rooms": rooms,
        "history": history,
        "gridded_image": gridded_image,
        "final_overlay": (
            draw_review_overlay(
                gridded_image,
                rooms,
            )
        ),
        "image_size": [
            width,
            height,
        ],
    }
