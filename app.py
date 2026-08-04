import base64
import hashlib
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from floorplan_editor import floorplan_editor
from outdoor_grouping import outdoor_grouping
from equipment import (
    DEFAULT_EQUIPMENT_FILENAME,
    load_vrv_equipment,
    recommend_indoor,
    recommend_outdoor,
)
from hvac import calculate_rows
from openai_gateway import analyze_floorplan
from pdf_utils import render_pdf_page


EQUIPMENT_FILE = (
    Path(__file__).resolve().parent
    / DEFAULT_EQUIPMENT_FILENAME
)


st.set_page_config(
    page_title="空調負荷計算",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --app-bg: #f5f7fb;
        --card-bg: #ffffff;
        --border: #dfe5ee;
        --text: #172033;
        --muted: #6b7280;
        --primary: #2563eb;
    }

    .stApp {
        background: var(--app-bg);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1800px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .app-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }

    .app-title {
        color: var(--text);
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: 0.01em;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
    }

    .section-label {
        color: var(--text);
        font-size: 0.96rem;
        font-weight: 750;
        margin: 0.1rem 0 0.55rem;
    }

    .soft-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .info-banner {
        background: #eef5ff;
        border: 1px solid #cfe0ff;
        border-radius: 9px;
        color: #31527d;
        font-size: 0.88rem;
        padding: 0.58rem 0.8rem;
        margin-bottom: 0.75rem;
    }

    .room-list {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.7rem;
        min-height: 180px;
    }

    .room-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.1rem;
        border-bottom: 1px solid #eef1f5;
        font-size: 0.88rem;
    }

    .room-row:last-child {
        border-bottom: none;
    }

    .room-dot {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex: 0 0 auto;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem;
        margin-top: 0.7rem;
    }

    .metric-caption {
        color: var(--muted);
        font-size: 0.78rem;
    }

    .metric-value {
        color: var(--primary);
        font-size: 1.3rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
    }

    div[data-testid="stDataEditor"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 650;
    }

    hr {
        border-color: #e8ecf2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()
    ).decode("ascii")


def demo_rooms():
    return [
        {
            "id": 1,
            "name": "區域1",
            "room_type": "一般辦公室",
            "confidence": 1.0,
            "color": "#ef4444",
            "unit_load": 120,
            "per_ping_load": 650,
            "included": True,
            "points": [
                {"x": 40, "y": 40},
                {"x": 190, "y": 40},
                {"x": 190, "y": 300},
                {"x": 40, "y": 300},
            ],
        },
        {
            "id": 2,
            "name": "區域2",
            "room_type": "一般辦公室",
            "confidence": 1.0,
            "color": "#f97316",
            "unit_load": 120,
            "per_ping_load": 650,
            "included": True,
            "points": [
                {"x": 40, "y": 310},
                {"x": 420, "y": 310},
                {"x": 420, "y": 650},
                {"x": 40, "y": 650},
            ],
        },
        {
            "id": 3,
            "name": "區域3",
            "room_type": "會議室",
            "confidence": 1.0,
            "color": "#f59e0b",
            "unit_load": 150,
            "per_ping_load": 800,
            "included": True,
            "points": [
                {"x": 720, "y": 40},
                {"x": 970, "y": 40},
                {"x": 970, "y": 290},
                {"x": 720, "y": 290},
            ],
        },
        {
            "id": 4,
            "name": "區域4",
            "room_type": "辦公室",
            "confidence": 1.0,
            "color": "#eab308",
            "unit_load": 120,
            "per_ping_load": 650,
            "included": True,
            "points": [
                {"x": 420, "y": 390},
                {"x": 720, "y": 390},
                {"x": 720, "y": 760},
                {"x": 420, "y": 760},
            ],
        },
    ]


st.markdown(
    """
    <div class="app-title-row">
      <div class="app-title">空調負荷計算</div>
    </div>
    <div class="app-subtitle">
      上傳平面圖、編輯空間框線，並依面積與每坪建議負荷值計算總熱負荷。
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 1. 上傳 PDF 圖面")
    uploaded = st.file_uploader(
        "選擇 PDF 或圖片",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 2. 辨識與計算設定")
    model = st.text_input("OpenAI 模型", value="gpt-5.6")
    pixels_per_m2 = st.number_input(
        "比例：每平方公尺像素數",
        min_value=1.0,
        value=10000.0,
        step=100.0,
    )


    page_no = 1
    run_ai = False
    load_demo = False

    if uploaded:
        data = uploaded.getvalue()

        if uploaded.name.lower().endswith(".pdf"):
            import fitz

            with fitz.open(stream=data, filetype="pdf") as doc:
                page_count = doc.page_count

            page_no = st.number_input(
                "PDF 頁碼",
                min_value=1,
                max_value=max(page_count, 1),
                value=1,
                step=1,
            )

        st.markdown("---")
        st.markdown("### 3. 編輯操作")
        run_ai = st.button(
            "AI 自動框選",
            type="primary",
            use_container_width=True,
        )
        load_demo = st.button(
            "載入測試框框",
            use_container_width=True,
        )
    else:
        st.info("請先上傳 PDF 或圖片。")


if uploaded:
    data = uploaded.getvalue()
    file_id = hashlib.sha256(data).hexdigest()

    if st.session_state.get("active_file_id") != file_id:
        st.session_state["active_file_id"] = file_id
        st.session_state["rooms"] = demo_rooms()
        st.session_state["editor_revision"] = (
            st.session_state.get("editor_revision", 0) + 1
        )

    if uploaded.name.lower().endswith(".pdf"):
        image, meta = render_pdf_page(data, int(page_no) - 1, 260)
        render_note = (
            f"建築裁切方式：{meta['method']}；"
            f"輸出：{meta['output_size_pixels'][0]} × "
            f"{meta['output_size_pixels'][1]} px"
        )
    else:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        render_note = f"圖片尺寸：{image.width} × {image.height} px"

    if run_ai:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

        try:
            result = analyze_floorplan(image, model=model)
            st.session_state["rooms"] = [
                room.model_dump() for room in result.rooms
            ]
            st.session_state["editor_revision"] = (
                st.session_state.get("editor_revision", 0) + 1
            )
            st.success(f"完成，共辨識 {len(result.rooms)} 個區域")
            if result.room_count_check:
                st.info(f"AI 自我檢查：{result.room_count_check}")
        except Exception as exc:
            st.error(str(exc))

    if load_demo:
        st.session_state["rooms"] = demo_rooms()
        st.session_state["editor_revision"] = (
            st.session_state.get("editor_revision", 0) + 1
        )

    st.markdown(
        """
        <div class="info-banner">
          提示：AI 已自動偵測房間區域。可在畫布中移動、拉伸或刪除框線；
          調整後，下方的面積與空調負荷會同步更新。
        </div>
        """,
        unsafe_allow_html=True,
    )

    rooms = st.session_state.get("rooms", demo_rooms())

    main_col, summary_col = st.columns([5.5, 1.25], gap="medium")

    with main_col:
        st.markdown(
            '<div class="section-label">圖面編輯區</div>',
            unsafe_allow_html=True,
        )
        st.caption(render_note)

        editor_value = floorplan_editor(
            image_data_url=to_data_url(image),
            rooms=rooms,
            zoom=0.60,
            revision=st.session_state.get("editor_revision", 0),
            key="floorplan_editor_main",
        )

    if isinstance(editor_value, dict) and "rooms" in editor_value:
        edited_rooms = editor_value["rooms"]
        st.session_state["rooms"] = edited_rooms
    else:
        edited_rooms = st.session_state.get("rooms", rooms)

    rows = calculate_rows(
        edited_rooms,
        image.width,
        image.height,
        pixels_per_m2,
    )
    df = pd.DataFrame(rows)

    expected_columns = [
        "編號",
        "區域名稱",
        "面積 (m²)",
        "面積 (坪)",
        "每坪建議負荷值 (kcal/h/坪)",
        "總熱負荷 (kW)",
        "室內機型號",
        "室內機數量",
        "室內機冷房能力 (kW)",
        "平均負荷 (kW/坪)",
        "室外機型號",
        "連結率 (%)",
    ]

    for column in expected_columns:
        if column not in df.columns:
            if column in (
                "面積 (m²)",
                "面積 (坪)",
                "總熱負荷 (kW)",
                "平均負荷 (kW/坪)",
            ):
                df[column] = 0.0
            elif column == "每坪建議負荷值 (kcal/h/坪)":
                df[column] = 650.0
            elif column == "室內機數量":
                df[column] = 1
            else:
                df[column] = ""

    df["面積 (坪)"] = (
        pd.to_numeric(df["面積 (m²)"], errors="coerce")
        .fillna(0.0)
        .div(3.305785)
        .round(2)
    )

    df["總熱負荷 (kW)"] = (
        pd.to_numeric(df["面積 (坪)"], errors="coerce").fillna(0.0)
        * pd.to_numeric(
            df["每坪建議負荷值 (kcal/h/坪)"],
            errors="coerce",
        ).fillna(650.0)
        / 860.0
    ).round(2)

    indoor_units = []
    outdoor_units = []

    try:
        indoor_units, outdoor_units = load_vrv_equipment(
            EQUIPMENT_FILE
        )
        st.sidebar.success(
            f"設備表已載入：室內機 {len(indoor_units)} 筆、"
            f"室外機 {len(outdoor_units)} 筆"
        )
    except FileNotFoundError:
        st.sidebar.warning(
            f"尚未找到 {DEFAULT_EQUIPMENT_FILENAME}；"
            "請將檔案放在 GitHub 專案根目錄。"
        )
    except Exception as exc:
        st.sidebar.error(f"設備報價單讀取失敗：{exc}")

    room_by_id_for_selection = {
        str(room.get("id")): room
        for room in st.session_state.get("rooms", edited_rooms)
    }

    # Auto-select the smallest indoor unit capacity above room demand.
    if indoor_units:
        for row_index, row in df.iterrows():
            room = room_by_id_for_selection.get(str(row["編號"]))
            if room is None:
                continue

            recommendation = recommend_indoor(
                float(row["總熱負荷 (kW)"]),
                indoor_units,
            )

            # Only auto-fill empty model fields, preserving user edits.
            if not room.get("indoor_model"):
                room["indoor_model"] = recommendation["model"]
                room["indoor_capacity_kw"] = recommendation["capacity_kw"]
                room["indoor_quantity"] = recommendation["quantity"]

            df.at[row_index, "室內機型號"] = (
                room.get("indoor_model") or ""
            )
            df.at[row_index, "室內機數量"] = int(
                room.get("indoor_quantity", 1) or 1
            )
            df.at[row_index, "室內機冷房能力 (kW)"] = (
                room.get("indoor_capacity_kw")
            )

    indoor_capacity_series = pd.to_numeric(
        df["室內機冷房能力 (kW)"],
        errors="coerce",
    )
    indoor_quantity_series = pd.to_numeric(
        df["室內機數量"],
        errors="coerce",
    ).fillna(1)

    df["平均負荷 (kW/坪)"] = (
        indoor_capacity_series
        * indoor_quantity_series
        / pd.to_numeric(df["面積 (坪)"], errors="coerce").replace(0, pd.NA)
    ).round(2)

    # 室外機分組：支援一層樓有一台或多台室外機，可用合併儲存格自由調整哪幾個房間
    # 屬於同一組，連結率／推薦室外機型號依分組即時重算。
    total_rooms = len(df)
    room_sig = tuple(str(v) for v in df["編號"].tolist()) if not df.empty else ()

    if st.session_state.get("outdoor_group_room_sig") != room_sig:
        st.session_state["outdoor_group_room_sig"] = room_sig
        st.session_state["outdoor_group_starts"] = {0} if total_rooms else set()
        st.session_state["outdoor_group_revision"] = (
            st.session_state.get("outdoor_group_revision", 0) + 1
        )

    group_starts = st.session_state.get("outdoor_group_starts", {0})

    def _build_groups(starts_sorted, total):
        groups = []
        for i, start in enumerate(starts_sorted):
            end = starts_sorted[i + 1] - 1 if i + 1 < len(starts_sorted) else total - 1
            groups.append((start, end))
        return groups

    outdoor_model_col = [""] * total_rooms
    connection_rate_col = [None] * total_rooms

    if outdoor_units and total_rooms:
        starts_sorted = sorted(group_starts)
        for (start, end) in _build_groups(starts_sorted, total_rooms):
            group_df = df.iloc[start:end + 1]
            indoor_rows_for_outdoor = [
                {
                    "indoor_model": str(row["室內機型號"]),
                    "indoor_quantity": int(row["室內機數量"] or 1),
                }
                for _, row in group_df.iterrows()
                if str(row["室內機型號"]).strip()
            ]
            rec = recommend_outdoor(
                indoor_rows_for_outdoor,
                outdoor_units,
                min_rate=105.0,
                max_rate=110.0,
            )
            model = rec.get("model") or ""
            rate = (
                round(rec["connection_rate"], 1)
                if rec.get("connection_rate") is not None
                else None
            )
            for i in range(start, end + 1):
                outdoor_model_col[i] = model
                connection_rate_col[i] = rate

    df["室外機型號"] = outdoor_model_col
    df["連結率 (%)"] = connection_rate_col

    for i, (_, row) in enumerate(df.iterrows()):
        room = room_by_id_for_selection.get(str(row["編號"]))
        if room is not None:
            room["outdoor_model"] = outdoor_model_col[i]
            room["connection_rate"] = connection_rate_col[i]

    if total_rooms:
        st.markdown('<div class="section-label">室外機分組</div>', unsafe_allow_html=True)
        st.caption(
            "拖曳虛線把手調整哪幾個房間屬於同一台室外機；點房間名稱可新增分組；"
            "點把手上方的 × 可合併。連結率會依分組即時重算。"
        )
        grouping_rows_payload = [
            {
                "index": i,
                "label": str(df.iloc[i]["區域名稱"]),
                "is_split": i in group_starts,
                "outdoor_model": outdoor_model_col[i],
                "connection_rate": connection_rate_col[i],
            }
            for i in range(total_rooms)
        ]
        grouping_value = outdoor_grouping(
            rows=grouping_rows_payload,
            revision=st.session_state.get("outdoor_group_revision", 0),
            key="outdoor_grouping_main",
        )
        if isinstance(grouping_value, dict) and "splits" in grouping_value:
            new_starts = {item["index"] for item in grouping_value["splits"]}
            new_starts.add(0)
            if new_starts != group_starts:
                st.session_state["outdoor_group_starts"] = new_starts
                st.rerun()

    st.session_state["rooms"] = list(
        room_by_id_for_selection.values()
    )
    df = df[expected_columns]

    total_area_m2 = (
        pd.to_numeric(df["面積 (m²)"], errors="coerce").fillna(0).sum()
        if not df.empty
        else 0
    )
    total_area_ping = (
        pd.to_numeric(df["面積 (坪)"], errors="coerce").fillna(0).sum()
        if not df.empty
        else 0
    )
    total_load = (
        pd.to_numeric(
            df["總熱負荷 (kW)"],
            errors="coerce",
        ).fillna(0).sum()
        if not df.empty
        else 0
    )

    with summary_col:
        st.markdown(
            '<div class="section-label">房間列表</div>',
            unsafe_allow_html=True,
        )

        if df.empty:
            st.info("尚無房間資料")
        else:
            palette = [
                "#ec4899",
                "#ef4444",
                "#f59e0b",
                "#eab308",
                "#22c55e",
                "#3b82f6",
                "#8b5cf6",
            ]
            with st.container(border=True):
                for idx, row in df.iterrows():
                    item_col, text_col = st.columns([0.22, 1.5])
                    with item_col:
                        st.markdown(
                            (
                                "<div style='width:28px;height:28px;"
                                "border-radius:7px;color:white;"
                                "display:flex;align-items:center;"
                                "justify-content:center;font-weight:700;"
                                f"background:{palette[idx % len(palette)]}'>"
                                f"{row['編號']}</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    with text_col:
                        st.markdown(
                            f"**{row['區域名稱']}**  \\n"
                            f"{row['面積 (m²)']:.2f} m²"
                        )
                    if idx < len(df) - 1:
                        st.divider()

        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-caption">總面積</div>
              <div class="metric-value">{total_area_m2:.2f} m²</div>
              <div class="metric-caption" style="margin-top:.35rem">
                {total_area_ping:.2f} 坪
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    result_title_col, export_col = st.columns([5, 1])

    with result_title_col:
        st.markdown(
            '<div class="section-label">空調負荷計算結果</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "可修改區域名稱、每坪建議負荷值、"
            "室內外機資料與連結率；面積與總熱負荷由系統自動計算。"
        )

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=[
            "編號",
            "面積 (m²)",
            "面積 (坪)",
            "總熱負荷 (kW)",
            "平均負荷 (kW/坪)",
            "室外機型號",
            "連結率 (%)",
        ],
        column_config={
            "編號": st.column_config.NumberColumn(
                "編號",
                disabled=True,
                format="%d",
                width="small",
            ),
            "區域名稱": st.column_config.TextColumn(
                "區域名稱",
                required=True,
                width="medium",
            ),
            "面積 (m²)": st.column_config.NumberColumn(
                "面積 (m²)",
                disabled=True,
                format="%.2f",
                width="small",
            ),
            "面積 (坪)": st.column_config.NumberColumn(
                "面積 (坪)",
                disabled=True,
                format="%.2f",
                width="small",
            ),
            "每坪建議負荷值 (kcal/h/坪)": (
                st.column_config.NumberColumn(
                    "每坪建議負荷值 (kcal/h/坪)",
                    min_value=0.0,
                    step=50.0,
                    format="%.0f",
                    required=True,
                    width="medium",
                )
            ),
            "總熱負荷 (kW)": st.column_config.NumberColumn(
                "總熱負荷 (kW)",
                disabled=True,
                format="%.2f",
                width="small",
            ),
            "室內機數量": st.column_config.NumberColumn(
                "室內機數量",
                min_value=1,
                step=1,
                format="%d",
                width="small",
            ),
            "平均負荷 (kW/坪)": st.column_config.NumberColumn(
                "平均負荷 (kW/坪)",
                disabled=True,
                format="%.2f",
                width="small",
            ),
            "室內機型號": st.column_config.TextColumn(
                "室內機型號",
                width="medium",
            ),
            "室內機冷房能力 (kW)": st.column_config.NumberColumn(
                "室內機冷房能力 (kW)",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                width="medium",
            ),
            "室外機型號": st.column_config.TextColumn(
                "室外機型號",
                disabled=True,
                width="medium",
                help="由上方「室外機分組」決定，不在這裡直接編輯",
            ),
            "連結率 (%)": st.column_config.NumberColumn(
                "連結率 (%)",
                disabled=True,
                format="%.1f",
                width="small",
                help="由上方「室外機分組」決定，不在這裡直接編輯",
            ),
        },
        key="hvac_result_editor",
    )

    def clean_optional_number(value):
        if pd.isna(value) or value in ("", "-"):
            return None
        return float(value)

    room_by_id = {
        str(room.get("id")): room
        for room in st.session_state.get("rooms", edited_rooms)
    }
    changed = False

    for _, row in edited_df.iterrows():
        room = room_by_id.get(str(row["編號"]))
        if room is None:
            continue

        updates = {
            "name": str(row["區域名稱"]).strip(),
            "per_ping_load": float(
                row["每坪建議負荷值 (kcal/h/坪)"]
            ),
            "indoor_model": (
                str(row["室內機型號"]).strip()
                if not pd.isna(row["室內機型號"])
                else ""
            ),
            "indoor_quantity": int(
                row["室內機數量"]
                if not pd.isna(row["室內機數量"])
                else 1
            ),
            "indoor_capacity_kw": clean_optional_number(
                row["室內機冷房能力 (kW)"]
            ),
        }

        for field, value in updates.items():
            if room.get(field) != value:
                room[field] = value
                changed = True

    if changed:
        st.session_state["rooms"] = list(room_by_id.values())
        if "hvac_result_editor" in st.session_state:
            del st.session_state["hvac_result_editor"]
        st.rerun()

    m1, m2, m3 = st.columns(3)
    m1.metric("總面積", f"{total_area_m2:.2f} m²")
    m2.metric("總坪數", f"{total_area_ping:.2f} 坪")
    m3.metric("總熱負荷", f"{total_load:,.2f} kW")

    with export_col:
        st.download_button(
            "匯出結果 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="空調負荷計算結果.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.markdown(
        """
        <div class="soft-card">
          <div class="section-label">開始使用</div>
          <div style="color:#6b7280">
            請從左側上傳 PDF 或圖片。上傳後，畫布、房間列表與空調負荷表格會顯示於此。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
