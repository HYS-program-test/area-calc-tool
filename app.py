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

from floorplan_detector import (
    DetectorConfig,
    crop_to_main_floorplan,
    detect_room_polygons,
)
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
from openai_room_detector import AIRoomDetectionOptions, detect_rooms_with_openai


st.set_page_config(
    page_title="平面圖空調設備選型",
    page_icon="❄️",
    layout="wide",
)

COLORS = [
    "#FF6347",
    "#3B82F6",
    "#22C55E",
    "#F59E0B",
    "#A855F7",
    "#06B6D4",
    "#EC4899",
    "#84CC16",
]
LOAD_OPTIONS = list(range(400, 1300, 100))


def init_session() -> None:
    defaults = {
        "file_key": None,
        "drawing": {"version": "4.4.0", "objects": []},
        "canvas_version": 0,
        "px_per_meter": None,
        "review": None,
        "ai_detection": None,
        "equipment_table": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def pdf_page(
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
def image_file(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_image(
    image: Image.Image,
    target_width: int,
) -> tuple[Image.Image, float]:
    scale = min(1.0, target_width / image.width)
    if scale == 1.0:
        return image.copy(), 1.0

    resized = image.resize(
        (
            round(image.width * scale),
            round(image.height * scale),
        ),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def safe_background(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def objects() -> list[dict]:
    return st.session_state.drawing.get("objects", [])


def records() -> list[dict]:
    output = []
    number = 0

    for object_index, obj in enumerate(objects()):
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
    stroke_width: int,
) -> dict:
    color = COLORS[(index - 1) % len(COLORS)]
    obj = polygon_to_fabric_path(
        room["points"],
        color,
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
    if len(row) > index and row[index]:
        return row[index].strip()
    return ""


@st.cache_data(show_spinner=False, ttl=300)
def equipment_data() -> tuple[list[str], dict[str, dict]]:
    try:
        credentials = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly"
            ],
        )
        client = gspread.authorize(credentials)
        sheet_id = st.secrets.get(
            "EQUIPMENT_SHEET_ID",
            "1hEt4uxBABBicxIMJuR57lMiigQYF02CQHZfB-Nc6vjo",
        )
        values = (
            client.open_by_key(sheet_id)
            .get_worksheet(0)
            .get_all_values()
        )

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


def export_pdf(
    image: Image.Image,
    room_records: list[dict],
    px_per_meter: float | None,
) -> bytes:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    for room in room_records:
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

        area_m2 = pixel_area_to_m2(
            polygon_area_px2(room["points"]),
            px_per_meter,
        )
        room_name = room.get("room_name") or room["room_id"]
        label = (
            room_name
            if area_m2 is None
            else f"{room_name} {area_m2:.2f}m2"
        )
        draw.text(
            (center_x, center_y),
            label,
            fill=room["color"],
        )

    buffer = io.BytesIO()
    output.save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


init_session()

st.markdown("## ❄️ AI 平面圖空調設備選型")
st.caption(
    "主要流程改為 OpenAI Vision 直接辨識空間候選框；"
    "OpenCV 僅保留為備援。AI 框線仍須人工確認後才能作為面積計算依據。"
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
    page_index = st.selectbox(
        "PDF 頁面",
        range(page_count),
        format_func=lambda index: f"第 {index + 1} 頁",
    )

api_key = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("圖面")
    dpi = st.slider(
        "PDF 解析度 DPI",
        120,
        300,
        200,
        20,
    )
    display_width = st.slider(
        "工作區寬度",
        700,
        1500,
        1100,
        50,
    )
    crop = st.checkbox(
        "自動裁切主要平面圖",
        True,
    )

    st.header("AI 辨識")
    ai_model = st.text_input(
        "OpenAI 視覺模型",
        value="gpt-4.1",
    )
    minimum_confidence = st.slider(
        "最低信心分數",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        format="%.2f",
    )
    include_balcony = st.checkbox("納入陽台", True)
    include_corridor = st.checkbox("納入走道／玄關", True)
    include_bathroom = st.checkbox("納入衛浴", True)
    include_stair = st.checkbox("納入樓梯", False)

    st.header("框線")
    tool = st.radio(
        "工具",
        [
            "選取／拖曳",
            "多邊形",
            "四角形",
            "校正線",
        ],
    )
    new_color = st.color_picker(
        "新增框線顏色",
        "#FF6347",
    )
    stroke_width = st.slider(
        "框線粗細",
        1,
        8,
        3,
    )

file_key = (
    f"{uploaded.name}:{len(data)}:{hash(data)}:"
    f"{page_index}:{dpi}:{display_width}:{crop}"
)
if st.session_state.file_key != file_key:
    st.session_state.file_key = file_key
    st.session_state.drawing = {
        "version": "4.4.0",
        "objects": [],
    }
    st.session_state.canvas_version += 1
    st.session_state.px_per_meter = None
    st.session_state.review = None
    st.session_state.ai_detection = None
    st.session_state.equipment_table = None

if is_pdf:
    image, auto_scale = pdf_page(
        data,
        page_index,
        dpi,
    )
else:
    image, auto_scale = image_file(data), None

if crop:
    image = crop_to_main_floorplan(image)

image, _ = resize_image(
    image,
    display_width,
)
image = safe_background(image)

button_ai, button_clear, button_fallback = st.columns(3)

with button_ai:
    if st.button(
        "✨ AI 直接辨識房間",
        type="primary",
        use_container_width=True,
        disabled=not api_key,
    ):
        options = AIRoomDetectionOptions(
            include_balcony=include_balcony,
            include_corridor=include_corridor,
            include_stair=include_stair,
            include_bathroom=include_bathroom,
            minimum_confidence=minimum_confidence,
        )

        try:
            with st.spinner(
                "OpenAI 正在辨識房間與產生候選多邊形，請稍候…"
            ):
                result = detect_rooms_with_openai(
                    api_key=api_key,
                    image=image.convert("RGB"),
                    model=ai_model,
                    options=options,
                )

            canvas_objects = [
                room_to_canvas_object(
                    room,
                    index,
                    stroke_width,
                )
                for index, room in enumerate(
                    result["rooms"],
                    start=1,
                )
            ]

            st.session_state.drawing = {
                "version": "4.4.0",
                "objects": canvas_objects,
            }
            st.session_state.ai_detection = result
            st.session_state.review = None
            st.session_state.canvas_version += 1
            st.rerun()

        except Exception as error:
            st.error(f"AI 辨識失敗：{error}")

with button_clear:
    if st.button(
        "清空全部框線",
        use_container_width=True,
    ):
        st.session_state.drawing = {
            "version": "4.4.0",
            "objects": [],
        }
        st.session_state.ai_detection = None
        st.session_state.review = None
        st.session_state.canvas_version += 1
        st.rerun()

with button_fallback:
    use_fallback = st.checkbox(
        "顯示 OpenCV 備援",
        value=False,
    )

if not api_key:
    st.error(
        "尚未設定 OPENAI_API_KEY，因此「AI 直接辨識房間」按鈕目前無法使用。"
    )

if use_fallback:
    with st.expander(
        "OpenCV 備援辨識設定",
        expanded=True,
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_area = st.number_input(
                "最小空間像素面積",
                1000,
                500000,
                12000,
                1000,
            )
            max_ratio = st.slider(
                "最大單一空間占比",
                0.10,
                0.90,
                0.35,
                0.05,
            )
        with col2:
            wall_length = st.slider(
                "牆線最短長度",
                8,
                100,
                30,
                2,
            )
            wall_thickness = st.slider(
                "牆線加粗",
                1,
                9,
                2,
            )
        with col3:
            door_gap = st.slider(
                "門洞封閉距離",
                5,
                100,
                12,
            )
            epsilon = st.number_input(
                "框線簡化程度",
                min_value=0.001,
                max_value=0.030,
                value=0.006,
                step=0.001,
                format="%.3f",
            )

        if st.button(
            "執行 OpenCV 備援辨識",
            use_container_width=True,
        ):
            config = DetectorConfig(
                wall_line_length=wall_length,
                wall_thickness=wall_thickness,
                door_gap_px=door_gap,
                min_room_area_px=min_area,
                max_room_area_ratio=max_ratio,
                polygon_epsilon_ratio=epsilon,
            )
            polygons, _ = detect_room_polygons(
                image.convert("RGB"),
                config,
            )
            fallback_objects = []
            for index, polygon in enumerate(polygons, start=1):
                obj = polygon_to_fabric_path(
                    polygon,
                    COLORS[(index - 1) % len(COLORS)],
                    stroke_width,
                    f"R{index:02d}",
                    "opencv",
                )
                if obj:
                    obj["room_name"] = ""
                    obj["room_type"] = ""
                    obj["confidence"] = None
                    obj["include_in_area"] = True
                    fallback_objects.append(obj)

            st.session_state.drawing = {
                "version": "4.4.0",
                "objects": fallback_objects,
            }
            st.session_state.ai_detection = None
            st.session_state.canvas_version += 1
            st.rerun()

# AI 結果摘要
if st.session_state.ai_detection:
    detection = st.session_state.ai_detection
    assessment = detection.get("image_assessment", {})

    st.markdown("### AI 辨識摘要")
    summary1, summary2, summary3 = st.columns(3)
    summary1.metric(
        "接受候選空間",
        len(detection.get("rooms", [])),
    )
    summary2.metric(
        "排除候選空間",
        len(detection.get("rejected_rooms", [])),
    )
    summary3.metric(
        "圖面品質",
        assessment.get("quality", "未知"),
    )

    if assessment.get("note"):
        st.info(assessment["note"])
    if detection.get("overall_note"):
        st.caption(detection["overall_note"])

    detected_table = pd.DataFrame(
        [
            {
                "編號": room["room_id"],
                "空間名稱": room["room_name"],
                "類型": room["room_type"],
                "納入面積": room["include_in_area"],
                "信心分數": round(room["confidence"], 2),
                "AI判斷": room["reason"],
            }
            for room in detection.get("rooms", [])
        ]
    )
    if not detected_table.empty:
        st.dataframe(
            detected_table,
            use_container_width=True,
            hide_index=True,
        )

    rejected = detection.get("rejected_rooms", [])
    if rejected:
        with st.expander("查看被程式排除的低信心候選框"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "空間名稱": room["room_name"],
                            "類型": room["room_type"],
                            "信心分數": round(
                                room["confidence"],
                                2,
                            ),
                            "排除原因": room.get(
                                "rejected_reason",
                                "",
                            ),
                        }
                        for room in rejected
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

mode = {
    "選取／拖曳": "transform",
    "多邊形": "polygon",
    "四角形": "rect",
    "校正線": "line",
}[tool]

st.markdown("### 候選空間人工確認")
st.caption(
    "AI 產生的是初始候選框。多邊形可重新繪製；"
    "選取模式可拖曳、拉伸及刪除。確認框線貼近內牆後，再進行比例尺校正。"
)

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_width,
    stroke_color=new_color,
    background_image=image,
    update_streamlit=True,
    height=image.height,
    width=image.width,
    drawing_mode=mode,
    initial_drawing=st.session_state.drawing,
    display_toolbar=True,
    key=f"canvas_{st.session_state.canvas_version}",
)

if canvas_result.json_data is not None:
    new_drawing = deepcopy(canvas_result.json_data)

    # streamlit drawable canvas 有時會遺失自訂 metadata，
    # 依物件順序將 AI 欄位補回。
    old_objects = objects()
    for index, obj in enumerate(
        new_drawing.get("objects", [])
    ):
        if index < len(old_objects):
            for metadata_key in [
                "room_id",
                "room_name",
                "room_type",
                "confidence",
                "include_in_area",
                "source",
                "ai_reason",
            ]:
                if metadata_key in old_objects[index]:
                    obj.setdefault(
                        metadata_key,
                        old_objects[index][metadata_key],
                    )

    st.session_state.drawing = new_drawing

room_records = records()

st.markdown("### 比例尺校正")
line_objects = [
    obj
    for obj in objects()
    if obj.get("type") == "line"
]
calibration1, calibration2, calibration3 = st.columns(3)

with calibration1:
    actual_cm = st.number_input(
        "最新校正線實際長度（cm）",
        min_value=1.0,
        value=1000.0,
    )
with calibration2:
    if st.button(
        "套用最新校正線",
        disabled=not line_objects,
        use_container_width=True,
    ):
        endpoints = fabric_line_endpoints(
            line_objects[-1]
        )
        st.session_state.px_per_meter = (
            px_per_meter_from_line(
                endpoints[0],
                endpoints[1],
                actual_cm / 100,
            )
        )
        st.rerun()
with calibration3:
    manual_px_per_meter = st.number_input(
        "或直接輸入 px/m",
        min_value=0.0,
        value=float(
            st.session_state.px_per_meter or 0
        ),
    )
    if manual_px_per_meter > 0:
        st.session_state.px_per_meter = (
            manual_px_per_meter
        )

if st.session_state.px_per_meter:
    st.success(
        f"目前比例尺："
        f"{st.session_state.px_per_meter:.3f} px/m"
    )
elif auto_scale:
    st.warning(
        f"偵測到圖面比例 1:{auto_scale}，"
        "仍建議使用圖上的已知尺寸線校正。"
    )
else:
    st.warning(
        "尚未校正比例尺，目前只能計算像素面積。"
    )

st.markdown("### 框線管理")
if room_records:
    selected = st.multiselect(
        "選擇空間",
        range(len(room_records)),
        format_func=lambda index: (
            f"{room_records[index]['room_id']}｜"
            f"{room_records[index].get('room_name') or '未命名'}｜"
            f"{room_records[index]['source']}"
        ),
    )

    manage1, manage2, manage3 = st.columns(3)
    with manage1:
        if st.button(
            "刪除選取空間",
            disabled=not selected,
            use_container_width=True,
        ):
            delete_indices = {
                room_records[index]["object_index"]
                for index in selected
            }
            st.session_state.drawing["objects"] = [
                obj
                for index, obj in enumerate(objects())
                if index not in delete_indices
            ]
            st.session_state.canvas_version += 1
            st.rerun()

    with manage2:
        replacement_color = st.color_picker(
            "選取空間的新顏色",
            "#3B82F6",
        )

    with manage3:
        if st.button(
            "套用顏色",
            disabled=not selected,
            use_container_width=True,
        ):
            for index in selected:
                object_index = room_records[index][
                    "object_index"
                ]
                st.session_state.drawing["objects"][
                    object_index
                ]["stroke"] = replacement_color
            st.session_state.canvas_version += 1
            st.rerun()

    st.markdown("#### 空間名稱與納入面積")
    metadata_df = pd.DataFrame(
        [
            {
                "編號": room["room_id"],
                "空間名稱": room.get("room_name", ""),
                "空間類型": room.get("room_type", ""),
                "納入面積": room.get(
                    "include_in_area",
                    True,
                ),
                "信心分數": room.get("confidence"),
            }
            for room in room_records
        ]
    )

    edited_metadata = st.data_editor(
        metadata_df,
        use_container_width=True,
        hide_index=True,
        disabled=["編號", "信心分數"],
        column_config={
            "納入面積": st.column_config.CheckboxColumn(
                "納入面積"
            ),
            "信心分數": st.column_config.NumberColumn(
                "信心分數",
                format="%.2f",
            ),
        },
        key="room_metadata_editor",
    )

    metadata_lookup = {
        row["編號"]: row
        for row in edited_metadata.to_dict("records")
    }
    for obj in objects():
        room_id = obj.get("room_id")
        if room_id in metadata_lookup:
            obj["room_name"] = metadata_lookup[room_id][
                "空間名稱"
            ]
            obj["room_type"] = metadata_lookup[room_id][
                "空間類型"
            ]
            obj["include_in_area"] = bool(
                metadata_lookup[room_id]["納入面積"]
            )

# 保留第二階段 AI 複核
st.markdown("### AI 二次複核")
if st.button(
    "請 OpenAI 再檢查目前人工調整後的框線",
    disabled=not room_records or not api_key,
):
    try:
        with st.spinner("OpenAI 正在複核目前框線…"):
            st.session_state.review = (
                review_room_candidates(
                    api_key,
                    image.convert("RGB"),
                    room_records,
                    ai_model,
                )
            )
    except Exception as error:
        st.error(f"OpenAI 複核失敗：{error}")

if st.session_state.review:
    st.dataframe(
        pd.DataFrame(
            st.session_state.review.get(
                "rooms",
                [],
            )
        ),
        use_container_width=True,
        hide_index=True,
    )
    if st.session_state.review.get(
        "missing_spaces"
    ):
        st.dataframe(
            pd.DataFrame(
                st.session_state.review[
                    "missing_spaces"
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.info(
        st.session_state.review.get(
            "overall_note",
            "",
        )
    )

st.markdown("### 面積與空調負荷")
load_per_ping = st.selectbox(
    "每坪建議負荷值（kcal/h·坪）",
    LOAD_OPTIONS,
    index=4,
)

area_rows = []
for room in records():
    area_px2 = polygon_area_px2(room["points"])
    area_m2 = pixel_area_to_m2(
        area_px2,
        st.session_state.px_per_meter,
    )
    load_result = cooling_load(
        area_m2,
        load_per_ping,
    )

    if not room.get("include_in_area", True):
        calculated_area_m2 = None
        calculated_ping = None
        kcal_h = None
        kw = None
    else:
        calculated_area_m2 = area_m2
        calculated_ping = load_result["ping"]
        kcal_h = load_result["kcal_h"]
        kw = load_result["kw"]

    area_rows.append(
        {
            "編號": room["room_id"],
            "空間名稱": room.get("room_name", ""),
            "空間類型": room.get("room_type", ""),
            "納入面積": room.get(
                "include_in_area",
                True,
            ),
            "面積(px²)": round(area_px2, 1),
            "面積(m²)": (
                round(calculated_area_m2, 2)
                if calculated_area_m2 is not None
                else None
            ),
            "面積(坪)": (
                round(calculated_ping, 2)
                if calculated_ping is not None
                else None
            ),
            "每坪建議負荷值": load_per_ping,
            "需求冷房能力(kcal/h)": (
                round(kcal_h)
                if kcal_h is not None
                else None
            ),
            "需求冷房能力(kW)": (
                round(kw, 2)
                if kw is not None
                else None
            ),
            "辨識來源": room["source"],
            "信心分數": room.get("confidence"),
        }
    )

area_df = pd.DataFrame(area_rows)
if not area_df.empty:
    st.dataframe(
        area_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("尚無空間資料。")

st.markdown("### 空調設備選型")
models, equipment_lookup = equipment_data()
equipment_rows = []

for row in area_rows:
    previous = next(
        (
            item
            for item in (
                st.session_state.equipment_table
                or []
            )
            if item.get("編號") == row["編號"]
        ),
        {},
    )
    indoor = previous.get("室內機", "")
    info = equipment_lookup.get(indoor, {})

    equipment_rows.append(
        {
            "編號": row["編號"],
            "空間名稱": previous.get(
                "空間名稱",
                row["空間名稱"],
            ),
            "面積(m²)": row["面積(m²)"] or 0,
            "每坪建議負荷值": previous.get(
                "每坪建議負荷值",
                load_per_ping,
            ),
            "需求冷房能力": (
                row["需求冷房能力(kcal/h)"] or 0
            ),
            "室內機": indoor,
            "類型": info.get("類型", ""),
            "室內機冷房能力": info.get(
                "室內機冷房能力",
                "",
            ),
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
        "編號": st.column_config.TextColumn(
            disabled=True
        ),
        "面積(m²)": st.column_config.NumberColumn(
            disabled=True
        ),
        "每坪建議負荷值": (
            st.column_config.SelectboxColumn(
                options=LOAD_OPTIONS
            )
        ),
        "需求冷房能力": (
            st.column_config.NumberColumn(
                disabled=True
            )
        ),
        "室內機": st.column_config.SelectboxColumn(
            options=models or [""]
        ),
        "類型": st.column_config.TextColumn(
            disabled=True
        ),
        "室內機冷房能力": (
            st.column_config.TextColumn(
                disabled=True
            )
        ),
        "室外機": st.column_config.TextColumn(
            disabled=True
        ),
    },
    key="equipment_editor",
)
st.session_state.equipment_table = (
    edited_equipment.to_dict("records")
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
        area_df.to_csv(index=False).encode(
            "utf-8-sig"
        ),
        f"{Path(uploaded.name).stem}_面積.csv",
        "text/csv",
        disabled=area_df.empty,
        use_container_width=True,
    )

with export3:
    st.download_button(
        "下載框選 PDF",
        export_pdf(
            image,
            records(),
            st.session_state.px_per_meter,
        ),
        f"{Path(uploaded.name).stem}_框面積.pdf",
        "application/pdf",
        use_container_width=True,
    )
