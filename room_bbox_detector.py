from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps
from openai import OpenAI


CANVAS_WIDTH = 1166
CANVAS_HEIGHT = 1200


@dataclass
class DetectionOptions:
    model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")
    max_retries: int = 2
    jpeg_quality: int = 94
    crop_padding_ratio: float = 0.035


BUILDING_PROMPT = """
你是建築平面圖裁切助手。

請只找出「主要建築平面配置圖」的外接矩形範圍，不要把以下內容納入：
- 道路、基地邊界、地界線
- 指北針、圖框、標題欄、註解
- 外圍尺寸標註及大片空白
- 與主要建築無關的附圖

座標以輸入圖片左上角為原點，輸出 0~1000 的整數相對座標。
裁切範圍應完整包含主要建築外牆，四周保留少量空間，不要切到牆。

只輸出 JSON：
{
  "building_bbox": {
    "x1": 0,
    "y1": 0,
    "x2": 1000,
    "y2": 1000
  },
  "confidence": 0.0
}
"""


SPACE_INVENTORY_PROMPT = """
你是一位建築平面圖判讀人員。這張圖片已經裁切到主要建築範圍。

本階段只做「空間盤點」，不要輸出任何座標，也不要框選物件。

請逐區閱讀整張圖，列出所有應被視為獨立室內使用面積的空間。
你要辨識的是完整空間，不是床、櫃子、樓梯、洗手台、門、門弧、尺寸線或設備。

判斷原則：
- 有完整牆體分隔的臥室、衛浴、儲藏室等，分別列出。
- 開放式客廳、餐廳、廚房若沒有完整隔牆，合併成一個空間。
- 樓梯本體、電梯設備、家具與櫃體不是房間。
- 陽台、露台、庭院、車道與室外空白不要列入。
- 走道若只是開放空間的一部分，不要切成多個碎片。
- 請依左上、上中、右上、左中、中央、右中、左下、中下、右下逐區檢查。
- 不確定的項目可以標示較低 confidence，但仍需避免把物件當空間。

只輸出 JSON：
{
  "spaces": [
    {
      "id": "S1",
      "name": "空間名稱",
      "location": "例如左下、右上、中央",
      "description": "以牆體與相鄰空間描述其位置",
      "confidence": 0.0
    }
  ]
}
"""


SINGLE_SPACE_BBOX_PROMPT_TEMPLATE = """
你是一位建築平面圖判讀人員。

目前只處理一個指定空間，不要框其他空間，也不要框任何家具或設備。

指定空間：
- id: {space_id}
- name: {space_name}
- location: {location}
- description: {description}

請在圖片中找到這個完整室內空間，輸出一個最能代表其主要可使用面積的矩形 bbox。

規則：
- 框的是整個空間，不是空間裡的床、櫃子、樓梯、設備、門或尺寸標註。
- bbox 邊界應儘量貼近牆內側完成面。
- 不得跨越實牆進入相鄰房間。
- 若空間為 L 型，先以能涵蓋主要使用區域、且不明顯跨牆的最大合理矩形表示。
- 若找不到此空間，found 請回傳 false。
- 座標以本張圖片左上角為原點，輸出 0~1000 的整數相對座標。

只輸出 JSON：
{
  "id": "{space_id}",
  "found": true,
  "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
  "confidence": 0.0,
  "check": "此框代表完整室內空間的簡短理由"
}
"""


REVIEW_PROMPT = """
你是一位負責檢查平面圖框選結果的建築師。

請檢查下面候選框是否確實代表完整室內空間，而不是家具、設備、樓梯、門、尺寸線或其他圖面物件。

候選框資料：
{candidate_json}

檢查規則：
- 完整房間或完整開放式使用空間才保留。
- 框到床、櫃子、樓梯、電梯設備、洗手台、門片、門弧、尺寸線者刪除。
- 明顯跨越實牆或跨入相鄰空間者刪除。
- 高度重複者只保留較合理的一個。
- 不要自行新增未在候選清單中的框。
- 座標維持 0~1000 相對座標。

只輸出 JSON：
{
  "rooms": [
    {
      "id": "S1",
      "name": "空間名稱",
      "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
      "confidence": 0.0
    }
  ]
}
"""


def _image_to_data_url(image: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=True,
    )
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
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
        raise ValueError("模型回傳內容不是有效 JSON。")

    return json.loads(text[start:end + 1])


def _clamp_1000(value: Any) -> int:
    try:
        return max(0, min(1000, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _normalize_bbox(raw: dict[str, Any]) -> dict[str, int]:
    x1 = _clamp_1000(raw.get("x1"))
    y1 = _clamp_1000(raw.get("y1"))
    x2 = _clamp_1000(raw.get("x2"))
    y2 = _clamp_1000(raw.get("y2"))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _call_vision_json(
    image: Image.Image,
    prompt: str,
    options: DetectionOptions,
) -> dict[str, Any]:
    client = OpenAI()
    data_url = _image_to_data_url(image, options.jpeg_quality)
    last_error: Exception | None = None

    for attempt in range(options.max_retries + 1):
        try:
            response = client.responses.create(
                model=options.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
            )
            return _extract_json(response.output_text)
        except Exception as exc:
            last_error = exc
            if attempt >= options.max_retries:
                break

    raise RuntimeError(f"OpenAI 影像分析失敗：{last_error}") from last_error


def resize_with_padding(
    image: Image.Image,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> tuple[Image.Image, dict[str, float]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    src_w, src_h = image.size

    scale = min(width / src_w, height / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))

    resized = image.resize(
        (resized_w, resized_h),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGB", (width, height), "white")
    offset_x = (width - resized_w) // 2
    offset_y = (height - resized_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas, {
        "source_width": src_w,
        "source_height": src_h,
        "canvas_width": width,
        "canvas_height": height,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "resized_width": resized_w,
        "resized_height": resized_h,
    }


def _relative_bbox_to_pixels(
    bbox: dict[str, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = round(bbox["x1"] / 1000 * width)
    y1 = round(bbox["y1"] / 1000 * height)
    x2 = round(bbox["x2"] / 1000 * width)
    y2 = round(bbox["y2"] / 1000 * height)
    return x1, y1, x2, y2


def detect_building_crop(
    original_image: Image.Image,
    options: DetectionOptions | None = None,
) -> dict[str, Any]:
    """
    第一次 OpenAI 呼叫：
    在整張原圖上找建築主體，回傳原圖像素 crop_box。
    """
    options = options or DetectionOptions()

    # 只為了送 API，先縮成較合理尺寸；座標仍可回推到原圖。
    preview, preview_transform = resize_with_padding(
        original_image,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
    )

    raw = _call_vision_json(preview, BUILDING_PROMPT, options)
    bbox = _normalize_bbox(raw.get("building_bbox", {}))

    # 相對座標先換成 preview canvas 座標。
    cx1, cy1, cx2, cy2 = _relative_bbox_to_pixels(
        bbox,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
    )

    # 去除補白 offset，再除以 scale，回推原圖座標。
    scale = preview_transform["scale"]
    offset_x = preview_transform["offset_x"]
    offset_y = preview_transform["offset_y"]

    ox1 = round((cx1 - offset_x) / scale)
    oy1 = round((cy1 - offset_y) / scale)
    ox2 = round((cx2 - offset_x) / scale)
    oy2 = round((cy2 - offset_y) / scale)

    original_w, original_h = original_image.size

    # 四周保留少量空間，模仿人工裁切時不貼死建築外牆。
    pad_x = round((ox2 - ox1) * options.crop_padding_ratio)
    pad_y = round((oy2 - oy1) * options.crop_padding_ratio)

    ox1 = max(0, ox1 - pad_x)
    oy1 = max(0, oy1 - pad_y)
    ox2 = min(original_w, ox2 + pad_x)
    oy2 = min(original_h, oy2 + pad_y)

    if ox2 - ox1 < 100 or oy2 - oy1 < 100:
        raise ValueError("建築裁切範圍過小，請檢查模型回傳結果。")

    return {
        "building_bbox_normalized": bbox,
        "crop_box_original": {
            "x1": ox1,
            "y1": oy1,
            "x2": ox2,
            "y2": oy2,
        },
        "confidence": float(raw.get("confidence", 0.5)),
    }


def crop_and_prepare_canvas(
    original_image: Image.Image,
    crop_result: dict[str, Any],
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """
    依第一次呼叫取得的 crop_box 裁切，再放大到固定畫布。
    """
    box = crop_result["crop_box_original"]
    crop_box = (box["x1"], box["y1"], box["x2"], box["y2"])

    cropped_image = original_image.crop(crop_box)
    canvas_image, resize_transform = resize_with_padding(
        cropped_image,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
    )

    coordinate_info = {
        "original_size": {
            "width": original_image.width,
            "height": original_image.height,
        },
        "crop_box_original": box,
        "crop_size": {
            "width": cropped_image.width,
            "height": cropped_image.height,
        },
        "canvas_size": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
        },
        "resize_transform": resize_transform,
    }

    return cropped_image, canvas_image, coordinate_info



def _bbox_iou(a: dict[str, int], b: dict[str, int]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _remove_duplicate_rooms(
    rooms: list[dict[str, Any]],
    iou_threshold: float = 0.72,
) -> list[dict[str, Any]]:
    """
    刪除高度重疊的重複框，保留信心較高者。
    """
    selected: list[dict[str, Any]] = []

    for room in sorted(
        rooms,
        key=lambda item: item.get("confidence", 0.0),
        reverse=True,
    ):
        bbox = room["bbox_canvas_normalized"]
        if any(
            _bbox_iou(bbox, kept["bbox_canvas_normalized"]) >= iou_threshold
            for kept in selected
        ):
            continue
        selected.append(room)

    return selected

def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _detect_space_inventory(
    canvas_image: Image.Image,
    options: DetectionOptions,
) -> list[dict[str, Any]]:
    raw = _call_vision_json(
        canvas_image,
        SPACE_INVENTORY_PROMPT,
        options,
    )

    spaces: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("spaces", [])):
        spaces.append(
            {
                "id": str(item.get("id") or f"S{index + 1}"),
                "name": str(item.get("name") or f"空間 {index + 1}"),
                "location": str(item.get("location") or ""),
                "description": str(item.get("description") or ""),
                "confidence": _safe_float(item.get("confidence"), 0.5),
            }
        )

    # 避免模型一次盤點過多可疑碎片。
    return spaces[:20]


def _detect_one_space_bbox(
    canvas_image: Image.Image,
    space: dict[str, Any],
    options: DetectionOptions,
) -> dict[str, Any] | None:
    prompt = SINGLE_SPACE_BBOX_PROMPT_TEMPLATE.format(
        space_id=space["id"],
        space_name=space["name"],
        location=space["location"],
        description=space["description"],
    )

    raw = _call_vision_json(canvas_image, prompt, options)

    if not bool(raw.get("found", True)):
        return None

    bbox = _normalize_bbox(raw.get("bbox", {}))
    width = bbox["x2"] - bbox["x1"]
    height = bbox["y2"] - bbox["y1"]

    # 過小或過度狹長者，通常不是完整室內空間。
    if width < 70 or height < 70:
        return None

    aspect_ratio = max(
        width / max(height, 1),
        height / max(width, 1),
    )
    if aspect_ratio > 5.0:
        return None

    return {
        "id": space["id"],
        "name": space["name"],
        "bbox_canvas_normalized": bbox,
        "confidence": min(
            space["confidence"],
            _safe_float(raw.get("confidence"), 0.5),
        ),
        "check": str(raw.get("check") or ""),
    }


def _review_candidate_rooms(
    canvas_image: Image.Image,
    candidates: list[dict[str, Any]],
    options: DetectionOptions,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    payload = [
        {
            "id": item["id"],
            "name": item["name"],
            "bbox": item["bbox_canvas_normalized"],
            "confidence": item["confidence"],
            "check": item.get("check", ""),
        }
        for item in candidates
    ]

    prompt = REVIEW_PROMPT.format(
        candidate_json=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    raw = _call_vision_json(canvas_image, prompt, options)

    reviewed: list[dict[str, Any]] = []
    for item in raw.get("rooms", []):
        bbox = _normalize_bbox(item.get("bbox", {}))
        width = bbox["x2"] - bbox["x1"]
        height = bbox["y2"] - bbox["y1"]

        if width < 70 or height < 70:
            continue

        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )
        if aspect_ratio > 5.0:
            continue

        reviewed.append(
            {
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or "室內空間"),
                "bbox_canvas_normalized": bbox,
                "confidence": _safe_float(item.get("confidence"), 0.5),
            }
        )

    return _remove_duplicate_rooms(reviewed)


def detect_rooms_on_cropped_canvas(
    canvas_image: Image.Image,
    options: DetectionOptions | None = None,
) -> dict[str, Any]:
    """
    多步驟 AI 辨識：
    1. 盤點有哪些室內空間，不輸出座標。
    2. 每個空間分別呼叫一次 API 取得 bbox。
    3. 再呼叫一次 API 審查候選框。
    """
    options = options or DetectionOptions()

    inventory = _detect_space_inventory(canvas_image, options)
    candidates: list[dict[str, Any]] = []

    for space in inventory:
        room = _detect_one_space_bbox(
            canvas_image,
            space,
            options,
        )
        if room is not None:
            candidates.append(room)

    reviewed = _review_candidate_rooms(
        canvas_image,
        candidates,
        options,
    )

    # Reviewer 若意外回傳空集合，保留初步結果，避免整頁無框。
    rooms = reviewed or _remove_duplicate_rooms(candidates)

    rooms.sort(
        key=lambda item: (
            item["bbox_canvas_normalized"]["y1"],
            item["bbox_canvas_normalized"]["x1"],
        )
    )

    for index, room in enumerate(rooms):
        room["id"] = chr(65 + index) if index < 26 else f"R{index + 1}"

    if not rooms:
        raise ValueError("AI 多步驟判讀後仍未找到有效室內空間。")

    return {
        "inventory": inventory,
        "candidates_before_review": candidates,
        "rooms": rooms,
    }

def canvas_normalized_bbox_to_canvas_pixels(
    bbox: dict[str, int],
) -> dict[str, float]:
    x1, y1, x2, y2 = _relative_bbox_to_pixels(
        bbox,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
    )
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    }


def canvas_point_to_original(
    x: float,
    y: float,
    coordinate_info: dict[str, Any],
) -> tuple[float, float]:
    """
    Canvas 座標 → 裁切圖座標 → 原圖座標。
    """
    transform = coordinate_info["resize_transform"]
    crop_box = coordinate_info["crop_box_original"]

    crop_x = (x - transform["offset_x"]) / transform["scale"]
    crop_y = (y - transform["offset_y"]) / transform["scale"]

    original_x = crop_x + crop_box["x1"]
    original_y = crop_y + crop_box["y1"]

    original_x = max(
        0.0,
        min(float(coordinate_info["original_size"]["width"]), original_x),
    )
    original_y = max(
        0.0,
        min(float(coordinate_info["original_size"]["height"]), original_y),
    )
    return original_x, original_y


def detection_to_polygons(
    room_detection: dict[str, Any],
    coordinate_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    同時保留：
    - points：裁切後固定 Canvas 座標，供頁面編輯器顯示
    - original_points：原始 PDF 圖片座標，供匯出疊圖
    """
    polygons: list[dict[str, Any]] = []

    for room in room_detection["rooms"]:
        b = canvas_normalized_bbox_to_canvas_pixels(
            room["bbox_canvas_normalized"]
        )

        canvas_points = [
            [b["x1"], b["y1"]],
            [b["x2"], b["y1"]],
            [b["x2"], b["y2"]],
            [b["x1"], b["y2"]],
        ]

        original_points = [
            list(canvas_point_to_original(x, y, coordinate_info))
            for x, y in canvas_points
        ]

        polygons.append(
            {
                "id": room["id"],
                "name": room["name"],
                "confidence": room["confidence"],
                "points": canvas_points,
                "original_points": original_points,
            }
        )

    return polygons


def run_same_flow_as_manual(
    original_image: Image.Image,
    options: DetectionOptions | None = None,
) -> dict[str, Any]:
    """
    完整流程：
    原圖 → 找建築主體 → 裁切 → 放大固定畫布
    → 框房間 → 產生 Canvas 與原圖兩套座標。
    """
    options = options or DetectionOptions()

    crop_result = detect_building_crop(original_image, options)
    cropped_image, canvas_image, coordinate_info = crop_and_prepare_canvas(
        original_image,
        crop_result,
    )
    room_detection = detect_rooms_on_cropped_canvas(canvas_image, options)
    polygons = detection_to_polygons(room_detection, coordinate_info)

    return {
        "crop_result": crop_result,
        "cropped_image": cropped_image,
        "canvas_image": canvas_image,
        "coordinate_info": coordinate_info,
        "room_detection": room_detection,
        "polygons": polygons,
    }
