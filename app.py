import base64
import hashlib
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from floorplan_editor import floorplan_editor
from hvac_table import hvac_table
from equipment import (
    DEFAULT_EQUIPMENT_FILENAME,
    load_equipment,
    recommend_indoor,
    recommend_outdoor,
    find_closest_outdoor_1to1,
    recommend_home_multi_outdoor,
)
from hvac import calculate_rows
from openai_gateway import analyze_floorplan
from pdf_utils import render_pdf_page


EQUIPMENT_FILE = (
    Path(__file__).resolve().parent
    / DEFAULT_EQUIPMENT_FILENAME
)

# 跟 frontend/index.html（floorplan_editor 改色按鈕）用的是同一組顏色，
# 圖面／房間列表／下方選機表三邊都靠這組顏色互相對應，不要單獨修改其中一邊。
ROOM_COLOR_PALETTE = [
    "#ef4444",
    "#f97316",
    "#f59e0b",
    "#22c55e",
    "#3b82f6",
    "#a855f7",
]


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
            "color": "#22c55e",
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
        if edited_rooms != st.session_state.get("rooms"):
            # 元件真的回傳了新資料（不是上一輪殘留的舊回傳值）才更新＋重新整理，
            # 這樣下一輪會用「剛剛更新過的最新資料」當作元件的顯示內容，
            # 不會再發生「這一輪傳給元件的還是更新前的舊資料，把剛剛的變更蓋回去」的問題。
            st.session_state["rooms"] = edited_rooms
            st.rerun()
    else:
        edited_rooms = st.session_state.get("rooms", rooms)

    rows = calculate_rows(
        edited_rooms,
        image.width,
        image.height,
        pixels_per_m2,
    )
    df = pd.DataFrame(rows)

    if "顏色" in df.columns:
        df["顏色"] = df["顏色"].apply(
            lambda c: c if c in ROOM_COLOR_PALETTE else ROOM_COLOR_PALETTE[0]
        )

    expected_columns = [
        "編號",
        "顏色",
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
            elif column == "顏色":
                df[column] = "#ef4444"
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
    home_indoor_units = []
    home_outdoor_units = []
    home_multi_indoor_units = []
    home_multi_outdoor_units = []
    commercial_indoor_units = []
    commercial_outdoor_units = []

    try:
        equipment = load_equipment(EQUIPMENT_FILE)
        indoor_units = equipment["vrv_indoor"]
        outdoor_units = equipment["vrv_outdoor"]
        home_indoor_units = equipment["home_indoor"]
        home_outdoor_units = equipment["home_outdoor"]
        home_multi_indoor_units = equipment["home_multi_indoor"]
        home_multi_outdoor_units = equipment["home_multi_outdoor"]
        commercial_indoor_units = equipment["commercial_indoor"]
        commercial_outdoor_units = equipment["commercial_outdoor"]
        st.sidebar.success(
            f"設備表已載入：VRV 室內機 {len(indoor_units)} 筆、室外機 {len(outdoor_units)} 筆；"
            f"家用一對一 室內機 {len(home_indoor_units)} 筆、室外機 {len(home_outdoor_units)} 筆；"
            f"家用一對多 室內機 {len(home_multi_indoor_units)} 筆、室外機 {len(home_multi_outdoor_units)} 筆；"
            f"商用一對一 室內機 {len(commercial_indoor_units)} 筆、室外機 {len(commercial_outdoor_units)} 筆"
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

    # 室外機配對：VRV 維持「相同顏色視為同一台室外機」＋連結率算法；
    # 家用一對一／商用一對一則是每間房間各自獨立配對（跟顏色分組無關，配對時也只會
    # 在同一個家族裡找——家用只配家用室外機、商用只配商用室外機，不會互相搭配）。
    # 家用一對多是「一台室外機接多台室內機」，邏輯上比較像 VRV，一樣按顏色分組配對
    # （室內機容量加總、找額定容量最接近的室外機，細節在 equipment.py 的
    # recommend_home_multi_outdoor）。
    # 顏色圖面／房間列表／下方選機表三邊互相同步（顏色的唯一真實來源是 room["color"]，
    # 這裡只是讀出來顯示）；室外機型號／連結率則是按下「確認」才會重新配對，不會即時自動變動。
    total_rooms = len(df)

    if "outdoor_match_results" not in st.session_state:
        st.session_state["outdoor_match_results"] = {}  # VRV：{顏色hex: {"model":str, "rate":float|None}}
    if "home_multi_match_results" not in st.session_state:
        st.session_state["home_multi_match_results"] = {}  # 家用一對多：{顏色hex: {"model":str}}
    if "oneone_match_results" not in st.session_state:
        st.session_state["oneone_match_results"] = {}  # 一對一：{編號str: {"model":str, "family":str}}
    match_results = st.session_state["outdoor_match_results"]
    home_multi_results = st.session_state["home_multi_match_results"]
    oneone_results = st.session_state["oneone_match_results"]

    outdoor_model_col = []
    connection_rate_col = []
    for _, row in df.iterrows():
        oneone_result = oneone_results.get(str(row["編號"]))
        if oneone_result:
            outdoor_model_col.append(oneone_result.get("model") or "")
            connection_rate_col.append(None)  # 一對一沒有連結率這個概念
            continue
        vrv_result = match_results.get(row["顏色"])
        if vrv_result:
            outdoor_model_col.append(vrv_result.get("model") or "")
            connection_rate_col.append(vrv_result.get("rate"))
            continue
        hm_result = home_multi_results.get(row["顏色"])
        if hm_result:
            outdoor_model_col.append(hm_result.get("model") or "")
            connection_rate_col.append(None)  # 家用一對多沒有連結率這個概念
            continue
        outdoor_model_col.append("")
        connection_rate_col.append(None)

    df["室外機型號"] = outdoor_model_col
    df["連結率 (%)"] = connection_rate_col

    for i, (_, row) in enumerate(df.iterrows()):
        room = room_by_id_for_selection.get(str(row["編號"]))
        if room is not None:
            room["outdoor_model"] = outdoor_model_col[i]
            room["connection_rate"] = connection_rate_col[i]

    def _classify_family(indoor_model: str) -> str | None:
        model_clean = indoor_model.strip().upper()
        if not model_clean:
            return None
        for u in indoor_units:
            if u["model"].strip().upper() == model_clean:
                return "vrv"
        for u in home_indoor_units:
            if u["model"].strip().upper() == model_clean:
                return "home"
        for u in home_multi_indoor_units:
            if u["model"].strip().upper() == model_clean:
                return "home_multi"
        for u in commercial_indoor_units:
            if u["model"].strip().upper() == model_clean:
                return "commercial"
        return None

    row_families = {
        str(r["編號"]): _classify_family(str(r["室內機型號"]))
        for _, r in df.iterrows()
    }

    # 合併儲存格的分組：家用/商用一對一每一列自己一組（永遠不跟別列合併）；
    # VRV 則是連續且顏色相同的列合併成一組。
    merge_group_ids: list[int] = []
    _current_gid = 0
    _prev_key = object()  # 保證第一列一定會開新的一組
    for _, r in df.iterrows():
        family = row_families.get(str(r["編號"]))
        is_oneone = family in ("home", "commercial")
        key = (str(r["編號"]), True) if is_oneone else (r["顏色"], False)
        if key != _prev_key:
            _current_gid += 1
        merge_group_ids.append(_current_gid)
        _prev_key = key

    if total_rooms:
        st.markdown('<div class="section-label">室外機配對</div>', unsafe_allow_html=True)
        st.caption(
            "圖面／房間列表／下方選機表的顏色互相同步（改任一邊都會連動其他兩邊）。"
            "VRV／家用一對多：相同顏色的房間視為接同一台室外機，一個負責算連結率、"
            "一個負責算室內機容量加總去配對。家用一對一／商用一對一：每間房間各自配對自己"
            "的室外機，跟顏色分組無關，也不會跨家用／商用互相搭配。顏色調整好之後按下面的"
            "「確認」，系統才會重新配對——改顏色本身不會自動觸發配對。"
        )
        if st.button("✅ 確認：配對室外機", key="confirm_outdoor_match_btn"):
            try:
                new_vrv_results = {}
                new_home_multi_results = {}
                new_oneone_results = {}

                # row_families 已經在上面算過了，這裡直接沿用（依室內機型號分類成 vrv / home / home_multi / commercial / 不明）

                # VRV：按顏色分組（只算被分類成 vrv 的那些列），沿用連結率邏輯
                if outdoor_units:
                    for color in df["顏色"].unique():
                        group_df = df[
                            (df["顏色"] == color)
                            & (df["編號"].astype(str).map(row_families) == "vrv")
                        ]
                        if group_df.empty:
                            continue
                        indoor_rows_for_outdoor = [
                            {
                                "indoor_model": str(r["室內機型號"]),
                                "indoor_quantity": int(r["室內機數量"]) if not pd.isna(r["室內機數量"]) else 1,
                            }
                            for _, r in group_df.iterrows()
                            if str(r["室內機型號"]).strip()
                        ]
                        rec = recommend_outdoor(
                            indoor_rows_for_outdoor,
                            outdoor_units,
                            indoor_units,
                            min_rate=105.0,
                            max_rate=110.0,
                        )
                        new_vrv_results[color] = {
                            "model": rec.get("model") or "",
                            "rate": (
                                round(rec["connection_rate"], 1)
                                if rec.get("connection_rate") is not None
                                else None
                            ),
                        }

                # 家用一對多：一台室外機接多台室內機，一樣按顏色分組，但用容量加總配對
                # （室內機容量加總可以略大於室外機額定容量，取最接近的那一台，見
                # equipment.py 的 recommend_home_multi_outdoor）
                if home_multi_outdoor_units:
                    for color in df["顏色"].unique():
                        group_df = df[
                            (df["顏色"] == color)
                            & (df["編號"].astype(str).map(row_families) == "home_multi")
                        ]
                        if group_df.empty:
                            continue
                        indoor_rows_for_outdoor = [
                            {
                                "indoor_model": str(r["室內機型號"]),
                                "indoor_quantity": int(r["室內機數量"]) if not pd.isna(r["室內機數量"]) else 1,
                            }
                            for _, r in group_df.iterrows()
                            if str(r["室內機型號"]).strip()
                        ]
                        rec = recommend_home_multi_outdoor(
                            indoor_rows_for_outdoor,
                            home_multi_outdoor_units,
                            home_multi_indoor_units,
                        )
                        new_home_multi_results[color] = {"model": rec.get("model") or ""}

                # 家用一對一／商用一對一：每一列各自獨立配對，不管顏色
                for _, r in df.iterrows():
                    family = row_families.get(str(r["編號"]))
                    indoor_model = str(r["室內機型號"]).strip()
                    if not indoor_model:
                        continue
                    if family == "home":
                        match = find_closest_outdoor_1to1(indoor_model, home_outdoor_units)
                        new_oneone_results[str(r["編號"])] = {"model": match.get("model") or "", "family": "home"}
                    elif family == "commercial":
                        match = find_closest_outdoor_1to1(indoor_model, commercial_outdoor_units)
                        new_oneone_results[str(r["編號"])] = {"model": match.get("model") or "", "family": "commercial"}

                st.session_state["outdoor_match_results"] = new_vrv_results
                st.session_state["home_multi_match_results"] = new_home_multi_results
                st.session_state["oneone_match_results"] = new_oneone_results

                # 除錯用摘要：不管配對結果好不好，至少讓你知道「這次真的有跑」以及跑出了什麼，
                # 不用再猜是按鈕沒反應還是配對結果剛好是空的。存進 session_state 是因為
                # 下面馬上就會 st.rerun()，一般的 st.success() 訊息在那之前就會被清掉看不到。
                vrv_matched = sum(1 for v in new_vrv_results.values() if v.get("model"))
                hm_matched = sum(1 for v in new_home_multi_results.values() if v.get("model"))
                oneone_matched = sum(1 for v in new_oneone_results.values() if v.get("model"))
                st.session_state["outdoor_match_summary"] = (
                    f"這次確認：VRV 配對成功 {vrv_matched}／{len(new_vrv_results)} 組顏色分組；"
                    f"家用一對多配對成功 {hm_matched}／{len(new_home_multi_results)} 組顏色分組；"
                    f"一對一（家用/商用）配對成功 {oneone_matched}／{len(new_oneone_results)} 間房間。"
                )

                # 同色的房間排在一起（穩定排序：同色內維持原本相對順序），
                # 房間列表跟下面的試算表都是從 st.session_state["rooms"] 這個順序畫出來的，
                # 所以排這裡兩邊會一起變。（一對一房間也會照顏色一起排到視覺上，
                # 但室外機配對本身跟顏色無關，純粹排序方便看。）
                sorted_df = df.sort_values("顏色", kind="stable")
                new_rooms_order = []
                for rid in sorted_df["編號"].tolist():
                    room = room_by_id_for_selection.get(str(rid))
                    if room is not None:
                        new_rooms_order.append(room)
                st.session_state["rooms"] = new_rooms_order

                st.rerun()
            except Exception as exc:
                st.error(f"配對室外機時發生錯誤：{exc}")

        if st.session_state.get("outdoor_match_summary"):
            st.caption(f"🔍 {st.session_state['outdoor_match_summary']}")

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
                                f"background:{row['顏色']}'>"
                                f"{row['編號']}</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    with text_col:
                        st.markdown(
                            f"**{row['區域名稱']}**  \n"
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

    def _capacity_candidates(units, demand_kw, low=0.85, high=1.8):
        """挑出容量大致適合這個熱負荷的機型（demand_kw 的 0.85~1.8 倍範圍內），
        範圍內完全沒有的話就退回容量最大的那一台頂著用。"""
        if not units or demand_kw <= 0:
            return []
        lo, hi = demand_kw * low, demand_kw * high
        cands = [u for u in units if lo <= u["capacity_kw"] <= hi]
        if not cands:
            cands = sorted(units, key=lambda u: u["capacity_kw"])[-1:]
        return sorted(cands, key=lambda u: u["capacity_kw"])

    def _build_model_options(demand_kw: float, locked_family: str | None):
        options = []
        if locked_family in (None, "vrv"):
            for u in _capacity_candidates(indoor_units, demand_kw):
                options.append({"value": u["model"], "label": f"VRV · {u['capacity_kw']:.1f}kW"})
        if locked_family in (None, "home"):
            for u in _capacity_candidates(home_indoor_units, demand_kw):
                options.append({"value": u["model"], "label": f"家用一對一 · {u['capacity_kw']:.1f}kW"})
        if locked_family in (None, "home_multi"):
            for u in _capacity_candidates(home_multi_indoor_units, demand_kw):
                options.append({"value": u["model"], "label": f"家用一對多 · {u['capacity_kw']:.1f}kW"})
        if locked_family in (None, "commercial"):
            for u in _capacity_candidates(commercial_indoor_units, demand_kw):
                options.append({"value": u["model"], "label": f"商用一對一 · {u['capacity_kw']:.1f}kW"})
        return options

    rows_payload = []
    for i, (_, row) in enumerate(df.iterrows()):
        room_id_str = str(row["編號"])

        # 防呆：這個房間所在的分組已經配對過室外機的話，室內機下拉選單就只能選
        # 同一個家族的機型；還沒配對過就顯示全部家族混合的選項。
        locked_family = None
        oneone_locked = oneone_results.get(room_id_str)
        if oneone_locked and oneone_locked.get("model"):
            locked_family = oneone_locked.get("family")
        elif match_results.get(row["顏色"], {}).get("model"):
            locked_family = "vrv"
        elif home_multi_results.get(row["顏色"], {}).get("model"):
            locked_family = "home_multi"

        demand_kw = float(row["總熱負荷 (kW)"]) if not pd.isna(row["總熱負荷 (kW)"]) else 0.0
        model_options = _build_model_options(demand_kw, locked_family)

        rows_payload.append({
            "room_id": int(row["編號"]),
            "name": str(row["區域名稱"]),
            "color": str(row["顏色"]),
            "area_m2": float(row["面積 (m²)"]),
            "area_ping": float(row["面積 (坪)"]),
            "per_ping_load": float(row["每坪建議負荷值 (kcal/h/坪)"]),
            "total_heat_kw": float(row["總熱負荷 (kW)"]),
            "indoor_model": "" if pd.isna(row["室內機型號"]) else str(row["室內機型號"]),
            "indoor_quantity": 1 if pd.isna(row["室內機數量"]) else int(row["室內機數量"]),
            "indoor_capacity_kw": (
                None if pd.isna(row["室內機冷房能力 (kW)"])
                else float(row["室內機冷房能力 (kW)"])
            ),
            "average_load": (
                None if pd.isna(row["平均負荷 (kW/坪)"])
                else float(row["平均負荷 (kW/坪)"])
            ),
            "outdoor_model": "" if pd.isna(row["室外機型號"]) else str(row["室外機型號"]),
            "connection_rate": (
                None if pd.isna(row["連結率 (%)"])
                else float(row["連結率 (%)"])
            ),
            "merge_group": merge_group_ids[i],
            "model_options": model_options,
        })

    table_value = hvac_table(
        rows=rows_payload,
        palette=ROOM_COLOR_PALETTE,
        revision=st.session_state.get("hvac_table_revision", 0),
        key="hvac_table_main",
    )

    if isinstance(table_value, dict) and table_value.get("updates"):
        nonce = table_value.get("nonce")
        if nonce is not None and nonce != st.session_state.get("last_hvac_table_nonce"):
            st.session_state["last_hvac_table_nonce"] = nonce

            room_by_id = {
                str(room.get("id")): room
                for room in st.session_state.get("rooms", edited_rooms)
            }
            changed = False

            for upd in table_value["updates"]:
                room = room_by_id.get(str(upd.get("room_id")))
                if room is None:
                    continue

                updates = {
                    "name": str(upd.get("name", "")).strip(),
                    "color": str(upd.get("color") or "#ef4444"),
                    "per_ping_load": float(upd.get("per_ping_load") or 0),
                    "indoor_model": str(upd.get("indoor_model", "")).strip(),
                    "indoor_quantity": int(upd.get("indoor_quantity") or 1),
                    "indoor_capacity_kw": (
                        float(upd["indoor_capacity_kw"])
                        if upd.get("indoor_capacity_kw") is not None
                        else None
                    ),
                }

                for field, value in updates.items():
                    if room.get(field) != value:
                        room[field] = value
                        changed = True

            if changed:
                st.session_state["rooms"] = list(room_by_id.values())
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
