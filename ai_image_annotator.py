from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

from openai import OpenAI
from PIL import Image


ANNOTATION_PROMPT = """
Edit this architectural floor plan image directly.

Preserve the original floor plan exactly:
- Do not redraw, simplify, move, erase, restyle, sharpen, blur, crop, rotate, or reinterpret any original wall, door, window, furniture, dimension, text, stair, elevator, or symbol.
- Do not change the image size or aspect ratio.
- The original drawing must remain visually unchanged.

Add only bright pure-red closed polygon outlines around the main usable indoor spaces.

Rules:
- Outline complete rooms or complete open-plan usable indoor spaces.
- Follow the inner face of walls.
- Use orthogonal polygon lines where the room boundary turns.
- For L-shaped rooms, draw an L-shaped polygon rather than a large rectangle.
- Do not outline furniture, beds, cabinets, sinks, toilets, stairs, elevators, doors, windows, dimension lines, text, balconies, terraces, driveways, outdoor areas, or empty exterior space.
- Do not fill the polygons.
- Use a solid pure red line RGB(255,0,0), approximately 6 pixels wide.
- Every outline must be closed.
- Do not add labels, numbers, arrows, legends, or any other marks.
- Return only the edited image.
"""


def _save_input_png(image: Image.Image) -> Path:
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = Path(temp.name)
    temp.close()
    image.convert("RGB").save(path, format="PNG")
    return path


def annotate_floorplan(
    image: Image.Image,
    model: str = "gpt-image-1",
) -> Image.Image:
    """
    將原圖送入 OpenAI Images Edit API，要求模型只增加紅色封閉框線。
    """
    input_path = _save_input_png(image)
    client = OpenAI()

    try:
        with input_path.open("rb") as image_file:
            result = client.images.edit(
                model=model,
                image=image_file,
                prompt=ANNOTATION_PROMPT,
                size="auto",
            )

        if not result.data:
            raise RuntimeError("OpenAI 沒有回傳圖片。")

        item = result.data[0]

        if getattr(item, "b64_json", None):
            image_bytes = base64.b64decode(item.b64_json)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if getattr(item, "url", None):
            raise RuntimeError(
                "目前程式預期 OpenAI 回傳 b64_json，但實際回傳 URL。"
                "請更新 SDK 或依目前 API 回傳格式調整下載流程。"
            )

        raise RuntimeError("OpenAI 回傳內容不含圖片資料。")
    finally:
        input_path.unlink(missing_ok=True)
