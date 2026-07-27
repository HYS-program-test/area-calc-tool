from __future__ import annotations

import base64
import io
import json
import re
from copy import deepcopy
from pathlib import Path

import fitz
import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from floorplan_detector import crop_to_main_floorplan
from geometry_utils import (
    cooling_load,
    fabric_object_points,
    is_area_object,
    pixel_area_to_m2,
    polygon_area_px2,
    polygon_to_fabric_polygon,
)
from openai_room_detector import (
    AIRoomDetectionOptions,
    detect_rooms_with_openai,
)


st.set_page_config(
    page_title="AI 平面圖空調設備選型",
    page_icon="❄️",
    layout="wide",
)

COLORS = [
    "#FF6347", "#3B82F6", "#22C55E", "#F59E0B",
    "#A855F7", "#06B6D4", "#EC4899", "#84CC16",
]
LOAD_OPTIONS = list(range(400, 1300, 100))
DEFAULT_DPI = 200
DEFAULT_WIDTH = 1100
DEFAULT_MODEL = "gpt-4.1"


def init_session() -> None:
    defaults = {
        "file_key": None,
        "drawing": {"version": "4.4.0", "objects": []},
        "canvas_version": 0,
        "px_per_meter": None,
        "ai_detection": None,
        "equipment_table": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def load_pdf_page(
    data: bytes,
    page_index: int,
    dpi: int,
) -> tuple[Image.Image, int | None]:
    document = fitz.open(stream=data, filetype="pdf")
    page = document.load_page(page_index)
    text = page.get_text()
    scale = None

    for pattern in [r"1\s*[:：]\s*(\d+)", r"1\s*/\s*(\d+)"]:
        match = re.search(pattern, text)
        if match and 10 <= int(match.group(1)) <= 5000:
            scale = int(match.group(1))
            break

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(dpi / 72, dpi / 72),
        alpha=False,
    )
    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )
    document.close()
    return image, scale


@st.cache_data(show_spinner=False)
def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_image(
    image: Image.Image,
    target_width: int,
) -> Image.Image:
    scale = min(1.0, target_width / image.width)
    if scale >= 1:
        return image.copy()

    return image.resize(
        (
            round(image.width * scale),
            round(image.height * scale),
        ),
        Image.Resampling.LANCZOS,
    )


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fabric_background_object(
    image: Image.Image,
) -> dict:
    """把底圖放進同一個 Fabric 畫布物件層。

    不再使用 st_canvas(background_image=...)，避免 rerun 後背景網址失效。
    """
    return {
        "type": "image",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": 0,
        "top": 0,
        "width": image.width,
        "height": image.height,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "opacity": 1,
        "visible": True,
        "selectable": False,
        "evented": False,
        "hasControls": False,
        "hasBorders": False,
        "objectCaching": False,
        "crossOrigin": "anonymous",
        "src": image_to_data_url(image),
        "name": "__floorplan_background__",
    }


def canvas_objects() -> list[dict]:
    return st.session_state.drawing.get("objects", [])


def ensure_background(
    image: Image.Image,
) -> None:
    foreground = [
        obj
        for obj in canvas_objects()
        if obj.get("name") != "__floorplan_background__"
    ]
    st.session_state.drawing = {
        "version": "4.4.0",
        "objects": [
            fabric_background_object(image),
            *foreground,
        ],
    }


def room_records() -> list[dict]:
    output = []
    number = 0

    for object_index, obj in enumerate(canvas_objects()):
        if obj.get("name") == "__floorplan_background__":
            continue
        if not is_area_object(obj):
            continue

        number += 1
        output.append(
            {
                "room_id": obj.get("room_id") or f"R{number:02d}",
                "room_name": obj.get("room_name", ""),
                "room_type": obj.get("room_type", ""),
                "confidence": obj.get("confidence"),
                "include_in_area": obj.get("include_in_area", True),
                "object_index": object_index,
                "points": fabric_object_points(obj),
                "color": obj.get(
                    "stroke",
                    COLORS[(number - 1) % len(COLORS)],
                ),
                "source": obj.get("source", "manual"),
            }
        )

    return output


def room_to_canvas_object(
    room: dict,
    index: int,
) -> dict:
    obj = polygon_to_fabric_polygon(
        room["points"],
        COLORS[(index - 1) % len(COLORS)],
        3,
        room.get("room_id") or f"R{index:02d}",
        "openai",
    )
    obj["room_name"] = room.get("room_name", "")
    obj["room_type"] = room.get("room_type", "")
    obj["confidence"] = room.get("confidence")
    obj["include_in_area"] = room.get("include_in_area", True)
    obj["ai_reason"] = room.get("reason", "")
    return obj


def _column(row: list[str], index: int) -> str:
    return row[index].strip() if len(row) > index and row[index] else ""


@st.cache_data(show_spinner=False, ttl=300)
def equipment_data() -> tuple[list[str], dict[str, dict]]:
    try:
        credentials = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        client = gspread.authorize(credentials)
        sheet_id = st.secrets.get(
            "EQUIPMENT_SHEET_ID",
            "1hEt4uxBABBicxIMJuR57lMiigQYF02CQHZfB-Nc6vjo",
        )
        values = client.open_by_key(sheet_id).get_worksheet(0).get_all_values()

        lookup = {}
        for row in values[2:]:
            indoor = _column(row, 3) or _column(row, 35)
            if indoor:
                lookup[indoor] = {
                    "類型": _column(row, 1),
                    "室外機": _column(row, 2),
                    "室內機冷房能力": _column(row, 16),
                }
        return sorted(lookup), lookup
    except Exception:
        return [], {}


def export_overlay_pdf(
    base_image: Image.Image,
    records: list[dict],
    px_per_meter: float | None,
) -> bytes:
    output = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    for room in records:
        points = [(round(x), round(y)) for x, y in room["points"]]
        if len(points) < 3:
            continue

        color = room["color"]
        draw.line(points + [points[0]], fill=color, width=5)

        area_m2 = pixel_area_to_m2(
            polygon_area_px2(room["points"]),
            px_per_meter,
        )
        room_name = room.get("room_name") or room["room_id"]
        label = (
            room_name
            if area_m2 is None
            else f"{room_name}  {area_m2:.2f} m²"
        )

        center_x = round(sum(x for x, _ in points) / len(points))
        center_y = round(sum(y for _, y in points) / len(points))
        draw.rectangle(
            (
                center_x - 4,
                center_y - 4,
                center_x + max(60, len(label) * 8),
                center_y + 18,
            ),
            fill="white",
            outline=color,
            width=2,
        )
        draw.text((center_x, center_y), label, fill=color)

    buffer = io.BytesIO()
    output.save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


init_session()

st.markdown("## ❄️ AI 平面圖空調設備選型")
st.caption(
    "畫面只保留一張可編輯平面圖。"
    "底圖、AI 框線、移動、拉伸與刪除都在同一個 Fabric 畫布中完成。"
)

uploaded = st.file_uploader(
    "上傳平面圖 PDF／PNG／JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)
if uploaded is None:
    st.info("請先上傳平面圖。")
    st.stop()

data = uploaded.getvalue()
is_pdf = uploaded.name.lower().endswith(".pdf")
page_index = 0

if is_pdf:
    document = fitz.open(stream=data, filetype="pdf")
    page_count = document.page_count
    document.close()

    if page_count > 1:
        page_index = st.selectbox(
            "PDF 頁面",
            range(page_count),
            format_func=lambda index: f"第 {index + 1} 頁",
        )

file_key = f"{uploaded.name}:{len(data)}:{hash(data)}:{page_index}"
if st.session_state.file_key != file_key:
    st.session_state.file_key = file_key
    st.session_state.drawing = {"version": "4.4.0", "objects": []}
    st.session_state.canvas_version += 1
    st.session_state.px_per_meter = None
    st.session_state.ai_detection = None
    st.session_state.equipment_table = None

if is_pdf:
    source_image, auto_scale = load_pdf_page(
        data,
        page_index,
        DEFAULT_DPI,
    )
else:
    source_image, auto_scale = load_image(data), None

source_image = crop_to_main_floorplan(source_image)
display_image = resize_image(source_image, DEFAULT_WIDTH)
ensure_background(display_image)

with st.sidebar:
    st.header("框線編輯")
    replacement_color = st.color_picker(
        "選取框線顏色",
        "#3B82F6",
    )
    st.caption(
        "在畫布點選框線即可拖曳或拉伸。"
        "刪除與改色請在畫布下方選取空間。"
    )

api_key = st.secrets.get("OPENAI_API_KEY", "")

action1, action2 = st.columns(2)

with action1:
    if st.button(
        "✨ AI 重新辨識房間",
        type="primary",
        use_container_width=True,
        disabled=not api_key,
    ):
        try:
            with st.spinner(
                "AI 先理解完整格局，再逐房間精修內牆邊界…"
            ):
                result = detect_rooms_with_openai(
                    api_key=api_key,
                    image=display_image.convert("RGB"),
                    model=DEFAULT_MODEL,
                    options=AIRoomDetectionOptions(),
                )

            st.session_state.drawing = {
                "version": "4.4.0",
                "objects": [
                    fabric_background_object(display_image),
                    *[
                        room_to_canvas_object(room, index)
                        for index, room in enumerate(
                            result["rooms"],
                            start=1,
                        )
                    ],
                ],
            }
            st.session_state.ai_detection = result
            st.session_state.canvas_version += 1
            st.rerun()
        except Exception as error:
            st.error(f"AI 辨識失敗：{error}")

with action2:
    if st.button(
        "清空全部框線",
        use_container_width=True,
    ):
        st.session_state.drawing = {
            "version": "4.4.0",
            "objects": [fabric_background_object(display_image)],
        }
        st.session_state.ai_detection = None
        st.session_state.canvas_version += 1
        st.rerun()

if not api_key:
    st.error("尚未設定 OPENAI_API_KEY。")

if st.session_state.ai_detection:
    detection = st.session_state.ai_detection
    c1, c2, c3 = st.columns(3)
    c1.metric("接受候選空間", len(detection.get("rooms", [])))
    c2.metric("排除候選空間", len(detection.get("rejected_rooms", [])))
    c3.metric(
        "圖面品質",
        detection.get("image_assessment", {}).get("quality", "未知"),
    )

st.markdown("### 候選空間人工確認")
st.caption(
    "畫布內的底圖不可選取；彩色框線可點選、拖曳、拉伸。"
)

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=3,
    stroke_color="#FF6347",
    background_color="#FFFFFF",
    update_streamlit=True,
    height=display_image.height,
    width=display_image.width,
    drawing_mode="transform",
    initial_drawing=st.session_state.drawing,
    display_toolbar=False,
    key=(
        f"single_canvas_{st.session_state.canvas_version}_"
        f"{display_image.width}x{display_image.height}"
    ),
)

if canvas_result.json_data is not None:
    new_drawing = deepcopy(canvas_result.json_data)
    old_objects = canvas_objects()

    for index, obj in enumerate(new_drawing.get("objects", [])):
        if index < len(old_objects):
            for metadata_key in [
                "name", "room_id", "room_name", "room_type",
                "confidence", "include_in_area", "source", "ai_reason",
            ]:
                if metadata_key in old_objects[index]:
                    obj.setdefault(
                        metadata_key,
                        old_objects[index][metadata_key],
                    )

    # 底圖一定維持最底層且不可操作。
    foreground = [
        obj
        for obj in new_drawing.get("objects", [])
        if obj.get("name") != "__floorplan_background__"
    ]
    st.session_state.drawing = {
        "version": "4.4.0",
        "objects": [
            fabric_background_object(display_image),
            *foreground,
        ],
    }

records = room_records()

st.markdown("### 框線管理")
if records:
    selected = st.multiselect(
        "選擇要改色或刪除的空間",
        range(len(records)),
        format_func=lambda index: (
            f"{records[index]['room_id']}｜"
            f"{records[index].get('room_name') or '未命名'}"
        ),
    )

    manage1, manage2 = st.columns(2)

    with manage1:
        if st.button(
            "套用選取顏色",
            disabled=not selected,
            use_container_width=True,
        ):
            for index in selected:
                object_index = records[index]["object_index"]
                st.session_state.drawing["objects"][object_index][
                    "stroke"
                ] = replacement_color
            st.session_state.canvas_version += 1
            st.rerun()

    with manage2:
        if st.button(
            "刪除選取框線",
            disabled=not selected,
            use_container_width=True,
        ):
            delete_indices = {
                records[index]["object_index"]
                for index in selected
            }
            st.session_state.drawing["objects"] = [
                obj
                for index, obj in enumerate(canvas_objects())
                if (
                    obj.get("name") == "__floorplan_background__"
                    or index not in delete_indices
                )
            ]
            st.session_state.canvas_version += 1
            st.rerun()

    metadata_df = pd.DataFrame(
        [
            {
                "編號": room["room_id"],
                "空間名稱": room.get("room_name", ""),
                "納入面積": room.get("include_in_area", True),
                "信心分數": room.get("confidence"),
            }
            for room in records
        ]
    )

    edited_metadata = st.data_editor(
        metadata_df,
        hide_index=True,
        use_container_width=True,
        disabled=["編號", "信心分數"],
        column_config={
            "納入面積": st.column_config.CheckboxColumn("納入面積"),
            "信心分數": st.column_config.NumberColumn(
                "信心分數",
                format="%.2f",
            ),
        },
        key="room_metadata",
    )

    metadata_lookup = {
        row["編號"]: row
        for row in edited_metadata.to_dict("records")
    }

    for obj in canvas_objects():
        room_id = obj.get("room_id")
        if room_id in metadata_lookup:
            obj["room_name"] = metadata_lookup[room_id]["空間名稱"]
            obj["include_in_area"] = bool(
                metadata_lookup[room_id]["納入面積"]
            )
else:
    st.info("尚無候選框線。")

st.markdown("### 比例尺與面積")
px_per_meter = st.number_input(
    "直接輸入比例尺（px/m）",
    min_value=0.0,
    value=float(st.session_state.px_per_meter or 0),
)
if px_per_meter > 0:
    st.session_state.px_per_meter = px_per_meter

load_per_ping = st.selectbox(
    "每坪建議負荷值（kcal/h·坪）",
    LOAD_OPTIONS,
    index=4,
)

area_rows = []
for room in room_records():
    area_px2 = polygon_area_px2(room["points"])
    area_m2 = pixel_area_to_m2(
        area_px2,
        st.session_state.px_per_meter,
    )
    load = cooling_load(area_m2, load_per_ping)
    included = room.get("include_in_area", True)

    area_rows.append(
        {
            "編號": room["room_id"],
            "空間名稱": room.get("room_name", ""),
            "納入面積": included,
            "面積(px²)": round(area_px2, 1),
            "面積(m²)": (
                round(area_m2, 2)
                if included and area_m2 is not None
                else None
            ),
            "面積(坪)": (
                round(load["ping"], 2)
                if included and load["ping"] is not None
                else None
            ),
            "需求冷房能力(kcal/h)": (
                round(load["kcal_h"])
                if included and load["kcal_h"] is not None
                else None
            ),
        }
    )

area_df = pd.DataFrame(area_rows)
if not area_df.empty:
    st.dataframe(
        area_df,
        hide_index=True,
        use_container_width=True,
    )

st.markdown("### 匯出")
export1, export2, export3 = st.columns(3)

with export1:
    st.download_button(
        "下載框線 JSON",
        json.dumps(
            st.session_state.drawing,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        f"{Path(uploaded.name).stem}_框線.json",
        "application/json",
        use_container_width=True,
    )

with export2:
    st.download_button(
        "下載面積 CSV",
        area_df.to_csv(index=False).encode("utf-8-sig"),
        f"{Path(uploaded.name).stem}_面積.csv",
        "text/csv",
        disabled=area_df.empty,
        use_container_width=True,
    )

with export3:
    st.download_button(
        "下載含底圖框面積 PDF",
        export_overlay_pdf(
            display_image,
            room_records(),
            st.session_state.px_per_meter,
        ),
        f"{Path(uploaded.name).stem}_含底圖框面積.pdf",
        "application/pdf",
        use_container_width=True,
    )
