from __future__ import annotations

import base64
import io
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw

DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")

PROMPT = """
你是一位建築平面圖判讀與 SVG 標註人員。

請直接閱讀輸入的建築平面圖，辨識主要室內使用空間，並輸出一份可疊加在原圖上的 SVG。

輸入圖片尺寸：
- width = {width}
- height = {height}

規則：
1. 根元素必須使用 width="{width}"、height="{height}"、viewBox="0 0 {width} {height}"。
2. SVG 內只能放房間框線，不得重畫底圖。
3. 每個室內空間只能使用一個 <polygon>。
4. points 必須使用輸入圖片的實際像素座標。
5. Polygon 要沿牆內側轉折；L 型空間使用 L 型 polygon。
6. 不得框家具、櫃體、衛浴設備、樓梯、電梯、門窗、文字、尺寸線、陽台、庭院、車道或建築外部。
7. 開放式客廳、餐廳、廚房若沒有完整隔牆，視為同一空間。
8. 有完整牆體分隔的臥室、衛浴、儲藏室，分別輸出。
9. Polygon 不得穿越實牆，也不得大幅重疊。
10. 每個 polygon 必須包含 id="room-A"、data-name="空間名稱"、fill="none"、stroke="#ff0000"、stroke-width="5"。
11. 不要輸出 path、rect、line、text、image 或其他元素。
12. 不要輸出 Markdown、JSON、說明或推理過程；只輸出完整 SVG。
"""

def _data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=95, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _extract_svg(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:svg|xml)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("<svg"), text.rfind("</svg>")
    if start < 0 or end < 0:
        raise ValueError("OpenAI 回覆中找不到完整 SVG。")
    return text[start:end + 6]

def _parse_points(value: str) -> list[list[float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", value or "")
    if len(nums) < 6 or len(nums) % 2:
        return []
    return [[float(nums[i]), float(nums[i + 1])] for i in range(0, len(nums), 2)]

def _area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0

def _validate(svg: str, width: int, height: int) -> dict[str, Any]:
    root = ET.fromstring(svg)
    rooms = []
    for element in root.iter():
        if element.tag.split("}")[-1] != "polygon":
            continue
        points = _parse_points(element.attrib.get("points", ""))
        clean = [[max(0.0, min(width - 1.0, x)), max(0.0, min(height - 1.0, y))] for x, y in points]
        area = _area(clean)
        if len(clean) < 3 or area <= 0:
            continue
        rooms.append({
            "id": element.attrib.get("id") or f"room-{len(rooms)+1}",
            "name": element.attrib.get("data-name") or "未命名空間",
            "points": clean,
            "area_pixels": area,
        })
    if not rooms:
        raise ValueError("SVG 中沒有可用的 polygon。")
    return {"svg": svg, "rooms": rooms, "image_width": width, "image_height": height}

def analyze_floorplan_as_svg(image: Image.Image, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    image = image.convert("RGB")
    width, height = image.size
    response = OpenAI().responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT.format(width=width, height=height)},
                {"type": "input_image", "image_url": _data_url(image), "detail": "high"},
            ],
        }],
    )
    svg = _extract_svg(response.output_text)
    result = _validate(svg, width, height)
    result["raw_output_text"] = response.output_text
    return result

def draw_svg_polygons(image: Image.Image, rooms: list[dict[str, Any]]) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for room in rooms:
        points = [(float(x), float(y)) for x, y in room["points"]]
        draw.line(points + [points[0]], fill=(255, 0, 0), width=5, joint="curve")
        x, y = points[0]
        draw.text((x + 7, y + 7), f'{room["id"]} {room["name"]}', fill=(255, 0, 0))
    return overlay
