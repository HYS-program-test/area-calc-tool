from __future__ import annotations

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
    fabric_line_endpoints,
    fabric_object_points,
    is_area_object,
    pixel_area_to_m2,
    polygon_area_px2,
    polygon_to_fabric_path,
    px_per_meter_from_line,
)
from openai_reviewer import review_room_candidates
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
        "review": None,
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


def canvas_background(image: Image.Image) -> Image.Image:
    """產生畫布專用、已完全載入的 RGB PNG。

    Canvas 套件對 RGBA 與新版 Streamlit 暫存圖片網址較敏感；
    本版固定使用 RGB PNG，並在 requirements.txt 鎖定相容版本。
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    buffer.seek(0)

    opened = Image.open(buffer)
    opened.load()
    return opened.convert("RGB").copy()


def canvas_objects() -> list[dict]:
    return st.session_state.drawing.get("objects", [])


def room_records() -> list[dict]:
    output = []
    number = 0

    for object_index, obj in enumerate(canvas_objects()):
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
    stroke_width: int = 3,
) -> dict:
    obj = polygon_to_fabric_path(
        room["points"],
        COLORS[(index - 1) % len(COLORS)],
        stroke_width,
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
    """將框線、名稱與面積直接疊加在原始底圖後輸出 PDF。"""
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

        polygon = points
        center_x = round(sum(x for x, _ in polygon) / len(polygon))
        center_y = round(sum(y for _, y in polygon) / len(polygon))

        text_box = draw.textbbox((0, 0), label)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        padding = 5

        draw.rectangle(
            (
                center_x - padding,
                center_y - padding,
                center_x + text_w + padding,
                center_y + text_h + padding,
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
    "OpenCV 先定位建築主體，再由 GPT 針對完整建築裁切圖辨識房間。"
    "AI 產生的框線可移動、拉伸、刪除或改色。"
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
    st.session_state.review = None
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
background = canvas_background(display_image)

# 側邊欄只保留人工編輯需要的項目。
with st.sidebar:
    st.header("框線編輯")
    edit_mode = st.radio(
        "操作",
        ["移動／拉伸", "刪除選取框線"],
    )
    replacement_color = st.color_picker(
        "選取框線顏色",
        "#3B82F6",
    )
    st.caption(
        "在畫布點選框線後，可拖曳移動或拉伸。"
        "刪除與改色請先在下方框線管理選取空間。"
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
        options = AIRoomDetectionOptions(
            include_balcony=True,
            include_corridor=True,
            include_stair=False,
            include_bathroom=True,
            minimum_confidence=0.35,
        )
        try:
            with st.spinner("OpenCV 定位建築主體，GPT 正在辨識房間…"):
                result = detect_rooms_with_openai(
                    api_key=api_key,
                    image=display_image.convert("RGB"),
                    model=DEFAULT_MODEL,
                    options=options,
                )

            st.session_state.drawing = {
                "version": "4.4.0",
                "objects": [
                    room_to_canvas_object(room, index)
                    for index, room in enumerate(result["rooms"], start=1)
                ],
            }
            st.session_state.ai_detection = result
            st.session_state.review = None
            st.session_state.canvas_version += 1
            st.rerun()
        except Exception as error:
            st.error(f"AI 辨識失敗：{error}")

with action2:
    if st.button(
        "清空全部框線",
        use_container_width=True,
    ):
        st.session_state.drawing = {"version": "4.4.0", "objects": []}
        st.session_state.ai_detection = None
        st.session_state.review = None
        st.session_state.canvas_version += 1
        st.rerun()

if not api_key:
    st.error("尚未設定 OPENAI_API_KEY。")

if st.session_state.ai_detection:
    detection = st.session_state.ai_detection
    assessment = detection.get("image_assessment", {})
    summary1, summary2, summary3 = st.columns(3)
    summary1.metric("接受候選空間", len(detection.get("rooms", [])))
    summary2.metric("排除候選空間", len(detection.get("rejected_rooms", [])))
    summary3.metric("圖面品質", assessment.get("quality", "未知"))

    accepted_points = [
        point
        for room in detection.get("rooms", [])
        for point in room.get("points", [])
    ]
    if accepted_points:
        min_x = min(point[0] for point in accepted_points)
        max_x = max(point[0] for point in accepted_points)
        min_y = min(point[1] for point in accepted_points)
        max_y = max(point[1] for point in accepted_points)
        st.caption(
            "AI 座標範圍："
            f"x={min_x:.1f}～{max_x:.1f}／{display_image.width}px，"
            f"y={min_y:.1f}～{max_y:.1f}／{display_image.height}px"
        )

    if detection.get("rejected_rooms"):
        with st.expander("查看被排除的候選框"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "空間名稱": item["room_name"],
                            "信心分數": item["confidence"],
                            "排除原因": item.get("rejected_reason", ""),
                        }
                        for item in detection["rejected_rooms"]
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )


def create_coordinate_diagnostic(
    base_image: Image.Image,
    current_records: list[dict],
) -> Image.Image:
    """直接用 PIL 畫出目前座標，不經 Fabric Canvas。

    若此圖正確但下方 Canvas 錯誤，代表問題位於 Fabric 座標層；
    若此圖也錯誤，代表 AI 回傳座標本身需要再改善。
    """
    preview = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)

    for room in current_records:
        points = [
            (round(x), round(y))
            for x, y in room["points"]
        ]
        if len(points) < 3:
            continue

        draw.line(
            points + [points[0]],
            fill=room["color"],
            width=4,
        )
        center_x = round(
            sum(x for x, _ in points) / len(points)
        )
        center_y = round(
            sum(y for _, y in points) / len(points)
        )
        draw.text(
            (center_x, center_y),
            room.get("room_name") or room["room_id"],
            fill=room["color"],
        )

    return preview


st.markdown("### 候選空間人工確認")
st.image(
    display_image,
    caption=f"底圖檢查：{display_image.width} × {display_image.height} px",
    use_container_width=True,
)
st.caption(
    "點選彩色框線後可拖曳、拉伸；"
    "框線刪除與改色可在畫布下方的管理區操作。"
)

current_records_before_canvas = room_records()

with st.expander(
    "座標驗證預覽（不經 Canvas）",
    expanded=True,
):
    st.image(
        create_coordinate_diagnostic(
            display_image,
            current_records_before_canvas,
        ),
        caption=(
            "此圖直接將目前多邊形座標畫在底圖上。"
            "先確認這裡的位置是否正確，再比較下方可編輯畫布。"
        ),
        use_container_width=True,
    )

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=3,
    stroke_color="#FF6347",
    background_image=background,
    background_color="#FFFFFF",
    update_streamlit=True,
    height=background.height,
    width=background.width,
    drawing_mode="transform",
    initial_drawing=st.session_state.drawing,
    display_toolbar=False,
    key=(
        f"canvas_{st.session_state.canvas_version}_"
        f"{background.width}x{background.height}"
    ),
)

if canvas_result.json_data is not None:
    new_drawing = deepcopy(canvas_result.json_data)
    old_objects = canvas_objects()

    for index, obj in enumerate(new_drawing.get("objects", [])):
        if index < len(old_objects):
            for metadata_key in [
                "room_id", "room_name", "room_type", "confidence",
                "include_in_area", "source", "ai_reason",
            ]:
                if metadata_key in old_objects[index]:
                    obj.setdefault(metadata_key, old_objects[index][metadata_key])

    st.session_state.drawing = new_drawing

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
                if index not in delete_indices
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
                "信心分數", format="%.2f"
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

st.markdown("### 比例尺校正")
st.caption("請在圖上已知尺寸的兩端建立校正線，或直接輸入 px/m。")

# 保留可建立校正線，但不放在左側調整項目。
calibration_mode = st.toggle("開啟校正線繪製", False)
if calibration_mode:
    calibration_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=3,
        stroke_color="#111111",
        background_image=background,
        background_color="#FFFFFF",
        update_streamlit=True,
        height=background.height,
        width=background.width,
        drawing_mode="line",
        display_toolbar=True,
        key="calibration_canvas",
    )
else:
    calibration_canvas = None

cal1, cal2, cal3 = st.columns(3)
with cal1:
    actual_cm = st.number_input(
        "校正線實際長度（cm）",
        min_value=1.0,
        value=1000.0,
    )
with cal2:
    can_apply = (
        calibration_canvas is not None
        and calibration_canvas.json_data
        and calibration_canvas.json_data.get("objects")
    )
    if st.button(
        "套用校正線",
        disabled=not can_apply,
        use_container_width=True,
    ):
        latest = calibration_canvas.json_data["objects"][-1]
        endpoints = fabric_line_endpoints(latest)
        if endpoints:
            st.session_state.px_per_meter = px_per_meter_from_line(
                endpoints[0],
                endpoints[1],
                actual_cm / 100,
            )
            st.rerun()
with cal3:
    manual_px = st.number_input(
        "直接輸入 px/m",
        min_value=0.0,
        value=float(st.session_state.px_per_meter or 0),
    )
    if manual_px > 0:
        st.session_state.px_per_meter = manual_px

if st.session_state.px_per_meter:
    st.success(f"目前比例尺：{st.session_state.px_per_meter:.3f} px/m")
elif auto_scale:
    st.warning(
        f"圖面文字可能包含比例 1:{auto_scale}；"
        "仍建議使用已知尺寸線校正。"
    )
else:
    st.warning("尚未完成比例尺校正，目前僅能計算像素面積。")

st.markdown("### 面積與空調負荷")
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
            "需求冷房能力(kW)": (
                round(load["kw"], 2)
                if included and load["kw"] is not None
                else None
            ),
        }
    )

area_df = pd.DataFrame(area_rows)
if not area_df.empty:
    st.dataframe(area_df, hide_index=True, use_container_width=True)

st.markdown("### 空調設備選型")
models, equipment_lookup = equipment_data()
equipment_rows = []

for row in area_rows:
    previous = next(
        (
            item for item in (st.session_state.equipment_table or [])
            if item.get("編號") == row["編號"]
        ),
        {},
    )
    indoor = previous.get("室內機", "")
    info = equipment_lookup.get(indoor, {})

    equipment_rows.append(
        {
            "編號": row["編號"],
            "空間名稱": previous.get("空間名稱", row["空間名稱"]),
            "面積(m²)": row["面積(m²)"] or 0,
            "每坪建議負荷值": previous.get(
                "每坪建議負荷值", load_per_ping
            ),
            "需求冷房能力": row["需求冷房能力(kcal/h)"] or 0,
            "室內機": indoor,
            "類型": info.get("類型", ""),
            "室內機冷房能力": info.get("室內機冷房能力", ""),
            "室外機": info.get("室外機", ""),
            "連結率": previous.get("連結率", ""),
        }
    )

equipment_df = pd.DataFrame(equipment_rows)
edited_equipment = st.data_editor(
    equipment_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "編號": st.column_config.TextColumn(disabled=True),
        "面積(m²)": st.column_config.NumberColumn(disabled=True),
        "每坪建議負荷值": st.column_config.SelectboxColumn(
            options=LOAD_OPTIONS
        ),
        "需求冷房能力": st.column_config.NumberColumn(disabled=True),
        "室內機": st.column_config.SelectboxColumn(
            options=models or [""]
        ),
        "類型": st.column_config.TextColumn(disabled=True),
        "室內機冷房能力": st.column_config.TextColumn(disabled=True),
        "室外機": st.column_config.TextColumn(disabled=True),
    },
    key="equipment_editor",
)
st.session_state.equipment_table = edited_equipment.to_dict("records")

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
