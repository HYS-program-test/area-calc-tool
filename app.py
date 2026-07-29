import base64
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
        {"id":1,"name":"區域1","room_type":"一般辦公室","confidence":1.0,"color":"#ef4444","unit_load":120,"included":True,
         "points":[{"x":40,"y":40},{"x":190,"y":40},{"x":190,"y":300},{"x":40,"y":300}]},
        {"id":2,"name":"區域2","room_type":"一般辦公室","confidence":1.0,"color":"#f97316","unit_load":120,"included":True,
         "points":[{"x":40,"y":310},{"x":420,"y":310},{"x":420,"y":650},{"x":40,"y":650}]},
        {"id":3,"name":"區域3","room_type":"會議室","confidence":1.0,"color":"#f59e0b","unit_load":150,"included":True,
         "points":[{"x":720,"y":40},{"x":970,"y":40},{"x":970,"y":290},{"x":720,"y":290}]},
        {"id":4,"name":"區域4","room_type":"辦公室","confidence":1.0,"color":"#eab308","unit_load":120,"included":True,
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
                st.session_state["rooms"] = [r.model_dump() for r in result.rooms]
                st.success(f"完成，共辨識 {len(result.rooms)} 個區域")
            except Exception as exc:
                st.error(str(exc))
    with col2:
        if st.button("載入測試框框"):
            st.session_state["rooms"] = demo_rooms()

    rooms = st.session_state.get("rooms", demo_rooms())
    editor_value = floorplan_editor(
        image_data_url=to_data_url(image),
        rooms=rooms,
        zoom=0.60,
        key="floorplan_editor_main",
    )

    edited_rooms = (editor_value or {}).get("rooms", rooms)
    st.session_state["rooms"] = edited_rooms

    rows = calculate_rows(edited_rooms, image.width, image.height, pixels_per_m2)
    df = pd.DataFrame(rows)

    st.subheader("空調負荷計算結果")
    st.dataframe(df, use_container_width=True, hide_index=True)

    included_df = df[df["納入計算"] == True] if not df.empty else df
    total_area = included_df["面積 (m²)"].sum() if not included_df.empty else 0
    total_load = included_df["總冷負荷 (W)"].sum() if not included_df.empty else 0
    total_rt = included_df["建議空調能力 (RT)"].sum() if not included_df.empty else 0

    m1,m2,m3 = st.columns(3)
    m1.metric("總面積", f"{total_area:.2f} m²")
    m2.metric("總冷負荷", f"{total_load:,.0f} W")
    m3.metric("總空調能力", f"{total_rt:.2f} RT")

    st.download_button(
        "匯出結果 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="空調負荷計算結果.csv",
        mime="text/csv",
    )
else:
    st.info("請先上傳 PDF 或圖片。")
