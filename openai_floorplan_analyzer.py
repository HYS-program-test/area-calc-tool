from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

from openai import OpenAI
from PIL import Image


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")


ANALYSIS_PROMPT = """
你是一位建築平面圖空間判讀人員。

請直接閱讀這張完整平面圖，辨識所有「由牆體界定的主要室內使用空間」，
並回傳每個空間貼近牆內側的 Polygon 座標。

重要定義：
- 你要框的是完整房間或完整開放式使用空間。
- 不是框床、桌椅、沙發、櫃體、流理台、洗手台、馬桶、樓梯、電梯設備、門片、窗戶、尺寸線或文字。
- 開放式客廳、餐廳與廚房若沒有完整隔牆，視為同一個空間。
- 有完整牆體分隔的臥室、衛浴、儲藏室等，分別框選。
- 不框陽台、露台、庭院、車道、道路、基地空白或建築外部。
- 門洞不代表空間中斷，應依門洞兩側牆體延伸判斷邊界。
- Polygon 應沿牆內側轉折，房間為 L 型時應輸出 L 型 Polygon，不要硬框成大矩形。
- 各 Polygon 不應穿越實牆，也不應重疊到其他房間。
- 請完整檢查左上、上中、右上、左中、中央、右中、左下、中下、右下，避免漏框。
- 座標以輸入圖片左上角為原點。
- 座標直接使用輸入圖片的實際像素座標，不要使用 0~1000 正規化座標。

比例尺：
- 若圖面有清楚尺寸線，請找出一條可信尺寸作為比例依據。
- 回傳該尺寸兩端在圖片中的像素座標及實際公尺長度。
- 若無法可靠判讀，scale_found 設為 false，不要猜測。

只輸出 JSON，不要輸出 Markdown、說明或推理過程。

格式：
{
  "image_width": 0,
  "image_height": 0,
  "scale": {
    "scale_found": false,
    "real_length_m": null,
    "pixel_start": null,
    "pixel_end": null,
    "source_text": null,
    "confidence": 0.0
  },
  "rooms": [
    {
      "id": "A",
      "name": "空間名稱",
      "polygon": [
        [0, 0],
        [0, 0],
        [0, 0]
      ],
      "confidence": 0.0
    }
  ]
}
"""


def _to_data_url(image: Image.Image, quality: int = 95) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("OpenAI 回傳內容中找不到 JSON。")

    return json.loads(text[start:end + 1])


def _clamp_point(
    point: list[Any],
    width: int,
    height: int,
) -> list[float] | None:
    if not isinstance(point, list) or len(point) != 2:
        return None

    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None

    return [
        max(0.0, min(float(width - 1), x)),
        max(0.0, min(float(height - 1), y)),
    ]


def _polygon_area_pixels(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0

    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1

    return abs(total) / 2.0


def _distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _validate_result(
    raw: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    scale_raw = raw.get("scale") or {}
    scale_found = bool(scale_raw.get("scale_found", False))

    pixel_start = _clamp_point(
        scale_raw.get("pixel_start"),
        image_width,
        image_height,
    )
    pixel_end = _clamp_point(
        scale_raw.get("pixel_end"),
        image_width,
        image_height,
    )

    try:
        real_length_m = float(scale_raw.get("real_length_m"))
    except (TypeError, ValueError):
        real_length_m = None

    pixels_per_meter = None
    if (
        scale_found
        and pixel_start is not None
        and pixel_end is not None
        and real_length_m is not None
        and real_length_m > 0
    ):
        pixel_length = _distance(pixel_start, pixel_end)
        if pixel_length > 0:
            pixels_per_meter = pixel_length / real_length_m
        else:
            scale_found = False
    else:
        scale_found = False

    rooms: list[dict[str, Any]] = []

    for index, room in enumerate(raw.get("rooms", [])):
        polygon_raw = room.get("polygon") or []
        polygon: list[list[float]] = []

        for point in polygon_raw:
            clean = _clamp_point(point, image_width, image_height)
            if clean is not None:
                polygon.append(clean)

        if len(polygon) < 3:
            continue

        area_pixels = _polygon_area_pixels(polygon)
        if area_pixels <= 0:
            continue

        try:
            confidence = float(room.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        area_m2 = None
        if pixels_per_meter:
            area_m2 = area_pixels / (pixels_per_meter ** 2)

        rooms.append(
            {
                "id": str(room.get("id") or chr(65 + index)),
                "name": str(room.get("name") or f"空間 {index + 1}"),
                "polygon": polygon,
                "points": polygon,
                "confidence": max(0.0, min(1.0, confidence)),
                "area_pixels": area_pixels,
                "area_m2": area_m2,
            }
        )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "scale": {
            "scale_found": scale_found,
            "real_length_m": real_length_m if scale_found else None,
            "pixel_start": pixel_start if scale_found else None,
            "pixel_end": pixel_end if scale_found else None,
            "pixels_per_meter": pixels_per_meter,
            "source_text": scale_raw.get("source_text"),
            "confidence": scale_raw.get("confidence", 0.0),
        },
        "rooms": rooms,
        "raw_response": raw,
    }


def analyze_floorplan(
    image: Image.Image,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    將完整平面圖圖片直接交給 OpenAI Vision 模型。

    AI 負責：
    - 理解空間
    - 決定房間數量
    - 回傳 Polygon
    - 嘗試辨識尺寸比例

    Python 僅負責：
    - 驗證座標
    - 計算 Polygon 面積
    """
    image = image.convert("RGB")
    width, height = image.size

    prompt = (
        ANALYSIS_PROMPT
        + f"\n輸入圖片實際尺寸為 width={width}, height={height}。"
        + "所有 polygon 與比例尺座標必須使用這個實際像素座標系統。"
    )

    client = OpenAI()
    response = client.responses.create(
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
                        "image_url": _to_data_url(image),
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    raw = _extract_json(response.output_text)
    return _validate_result(raw, width, height)
