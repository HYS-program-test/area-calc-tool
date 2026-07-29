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

st.set_page_config(page_title="空調負荷計算", layout="wide")
st.title("空調負荷計算")

def to_data_url(image: Image.Image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def demo_rooms():
    return [
        {"id":1,"name":"區域1","room_type":"一般辦公室","confidence":1.0,"color":"#ef4444","unit_load":120,"per_ping_load":650,"included":True,
         "points":[{"x":40,"y":40},{"x":190,"y":40},{"x":190,"y":300},{"x":40,"y":300}]},
        {"id":2,"name":"區域2","room_type":"一般辦公室","confidence":1.0,"color":"#f97316","unit_load":120,"per_ping_load":650,"included":True,
         "points":[{"x":40,"y":310},{"x":420,"y":310},{"x":420,"y":650},{"x":40,"y":650}]},
        {"id":3,"name":"區域3","room_type":"會議室","confidence":1.0,"color":"#f59e0b","unit_load":150,"per_ping_load":800,"included":True,
         "points":[{"x":720,"y":40},{"x":970,"y":40},{"x":970,"y":290},{"x":720,"y":290}]},
        {"id":4,"name":"區域4","room_type":"辦公室","confidence":1.0,"color":"#eab308","unit_load":120,"per_ping_load":650,"included":True,
         "points":[{"x":420,"y":390},{"x":720,"y":390},{"x":720,"y":760},{"x":420,"y":760}]},
    ]

uploaded = st.file_uploader(
    "上傳平面圖 PDF 或圖片",
    type=["pdf","png","jpg","jpeg"],
)

model = st.text_input("OpenAI 模型", value="gpt-5.6")
pixels_per_m2 = st.number_input("比例：每平方公尺像素數", min_value=1.0, value=10000.0, step=100.0)

if uploaded:
    data = uploaded.getvalue()
    file_id = hashlib.sha256(data).hexdigest()

    # Reset polygons only when the uploaded source file actually changes.
    if st.session_state.get("active_file_id") != file_id:
        st.session_state["active_file_id"] = file_id
        st.session_state["rooms"] = demo_rooms()
        st.session_state["editor_revision"] = (
            st.session_state.get("editor_revision", 0) + 1
        )
    if uploaded.name.lower().endswith(".pdf"):
        import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
        page_no = st.number_input("PDF 頁碼", 1, max(page_count,1), 1)
        image, meta = render_pdf_page(data, int(page_no)-1, 220)
        st.caption(f"建築裁切方式：{meta['method']}；輸出：{meta['output_size_pixels'][0]}×{meta['output_size_pixels'][1]} px")
    else:
        image = Image.open(io.BytesIO(data)).convert("RGB")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("OpenAI 自動框選", type="primary"):
            if "OPENAI_API_KEY" in st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            try:
                result = analyze_floorplan(image, model=model)
                st.session_state["rooms"] = [
                    r.model_dump() for r in result.rooms
                ]
                st.session_state["editor_revision"] = (
                    st.session_state.get("editor_revision", 0) + 1
                )
                st.success(f"完成，共辨識 {len(result.rooms)} 個區域")
            except Exception as exc:
                st.error(str(exc))
    with col2:
        if st.button("載入測試框框"):
            st.session_state["rooms"] = demo_rooms()
            st.session_state["editor_revision"] = (
                st.session_state.get("editor_revision", 0) + 1
            )

    rooms = st.session_state.get("rooms", demo_rooms())
    # floorplan_editor.py must include the `revision` parameter.
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

    # 防止任何欄位缺失造成 KeyError。
    for column in expected_columns:
        if column not in df.columns:
            if column in ("面積 (m²)", "面積 (坪)", "總熱負荷 (kcal/h)"):
                df[column] = 0.0
            elif column == "每坪建議負荷值 (kcal/h/坪)":
                df[column] = 650.0
            else:
                df[column] = ""

    # 再次強制依公式計算，確保畫面一定有「面積 (坪)」與正確總熱負荷。
    df["面積 (坪)"] = (
        pd.to_numeric(df["面積 (m²)"], errors="coerce").fillna(0.0)
        / 3.305785
    ).round(2)

    df["總熱負荷 (kcal/h)"] = (
        pd.to_numeric(df["面積 (坪)"], errors="coerce").fillna(0.0)
        * pd.to_numeric(
            df["每坪建議負荷值 (kcal/h/坪)"],
            errors="coerce",
        ).fillna(650.0)
    ).round(2)

    df = df[expected_columns]

    st.subheader("空調負荷計算結果")
    st.caption(
        "可直接修改區域名稱、空間類型、每坪建議負荷值、"
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
                "編號", disabled=True, format="%d"
            ),
            "區域名稱": st.column_config.TextColumn(
                "區域名稱", required=True
            ),
            "面積 (m²)": st.column_config.NumberColumn(
                "面積 (m²)", disabled=True, format="%.2f"
            ),
            "面積 (坪)": st.column_config.NumberColumn(
                "面積 (坪)", disabled=True, format="%.2f"
            ),
            "空間類型": st.column_config.SelectboxColumn(
                "空間類型",
                options=[
                    "一般辦公室", "辦公室", "主管室", "會議室",
                    "教室", "商店", "機房", "走道", "其他",
                ],
                required=True,
            ),
            "每坪建議負荷值 (kcal/h/坪)": st.column_config.NumberColumn(
                "每坪建議負荷值 (kcal/h/坪)",
                min_value=0.0,
                step=50.0,
                format="%.0f",
                required=True,
            ),
            "總熱負荷 (kcal/h)": st.column_config.NumberColumn(
                "總熱負荷 (kcal/h)",
                disabled=True,
                format="%.2f",
            ),
            "室內機型號": st.column_config.TextColumn("室內機型號"),
            "室內機冷房能力 (kW)": st.column_config.NumberColumn(
                "室內機冷房能力 (kW)",
                min_value=0.0,
                step=0.1,
                format="%.2f",
            ),
            "室外機型號": st.column_config.TextColumn("室外機型號"),
            "連結率 (%)": st.column_config.NumberColumn(
                "連結率 (%)",
                min_value=0.0,
                step=1.0,
                format="%.1f",
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

    total_area_m2 = df["面積 (m²)"].sum() if not df.empty else 0
    total_area_ping = df["面積 (坪)"].sum() if not df.empty else 0
    total_load = (
        df["總熱負荷 (kcal/h)"].sum()
        if not df.empty
        else 0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("總面積", f"{total_area_m2:.2f} m²")
    m2.metric("總坪數", f"{total_area_ping:.2f} 坪")
    m3.metric("總熱負荷", f"{total_load:,.2f} kcal/h")

    st.download_button(
        "匯出結果 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="空調負荷計算結果.csv",
        mime="text/csv",
    )
else:
    st.info("請先上傳 PDF 或圖片。")
