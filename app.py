import base64
import hashlib
import io
import os

import pandas as pd
import streamlit as st
from PIL import Image

from floorplan_editor import floorplan_editor
from hvac import calculate_rows
from openai_gateway import analyze_floorplan
from pdf_utils import render_pdf_page


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
        image, meta = render_pdf_page(data, int(page_no) - 1, 220)
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
        "空間類型",
        "每坪建議負荷值 (kcal/h/坪)",
        "總熱負荷 (kcal/h)",
        "室內機型號",
        "室內機冷房能力 (kW)",
        "室外機型號",
        "連結率 (%)",
    ]

    for column in expected_columns:
        if column not in df.columns:
            if column in (
                "面積 (m²)",
                "面積 (坪)",
                "總熱負荷 (kcal/h)",
            ):
                df[column] = 0.0
            elif column == "每坪建議負荷值 (kcal/h/坪)":
                df[column] = 650.0
            else:
                df[column] = ""

    df["面積 (坪)"] = (
        pd.to_numeric(df["面積 (m²)"], errors="coerce")
        .fillna(0.0)
        .div(3.305785)
        .round(2)
    )

    df["總熱負荷 (kcal/h)"] = (
        pd.to_numeric(df["面積 (坪)"], errors="coerce").fillna(0.0)
        * pd.to_numeric(
            df["每坪建議負荷值 (kcal/h/坪)"],
            errors="coerce",
        ).fillna(650.0)
    ).round(2)

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
            df["總熱負荷 (kcal/h)"],
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
            room_html = ['<div class="room-list">']
            for idx, row in df.iterrows():
                color = palette[idx % len(palette)]
                room_html.append(
                    f"""
                    <div class="room-row">
                      <span class="room-dot" style="background:{color}">
                        {row['編號']}
                      </span>
                      <span>
                        {row['區域名稱']}
                        <br>
                        <small>{row['面積 (m²)']:.2f} m²</small>
                      </span>
                    </div>
                    """
                )
            room_html.append("</div>")
            st.markdown("".join(room_html), unsafe_allow_html=True)

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
            "可修改區域名稱、空間類型、每坪建議負荷值、"
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
            "總熱負荷 (kcal/h)",
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
            "空間類型": st.column_config.SelectboxColumn(
                "空間類型",
                options=[
                    "一般辦公室",
                    "辦公室",
                    "主管室",
                    "會議室",
                    "教室",
                    "商店",
                    "機房",
                    "走道",
                    "其他",
                ],
                required=True,
                width="medium",
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
            "總熱負荷 (kcal/h)": st.column_config.NumberColumn(
                "總熱負荷 (kcal/h)",
                disabled=True,
                format="%.2f",
                width="medium",
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
                width="medium",
            ),
            "連結率 (%)": st.column_config.NumberColumn(
                "連結率 (%)",
                min_value=0.0,
                step=1.0,
                format="%.1f",
                width="small",
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
            "room_type": str(row["空間類型"]),
            "per_ping_load": float(
                row["每坪建議負荷值 (kcal/h/坪)"]
            ),
            "indoor_model": (
                str(row["室內機型號"]).strip()
                if not pd.isna(row["室內機型號"])
                else ""
            ),
            "indoor_capacity_kw": clean_optional_number(
                row["室內機冷房能力 (kW)"]
            ),
            "outdoor_model": (
                str(row["室外機型號"]).strip()
                if not pd.isna(row["室外機型號"])
                else ""
            ),
            "connection_rate": clean_optional_number(
                row["連結率 (%)"]
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
    m3.metric("總熱負荷", f"{total_load:,.2f} kcal/h")

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
