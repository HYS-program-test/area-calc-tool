from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import fitz
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from openai_svg_analyzer import analyze_floorplan_as_svg, draw_svg_polygons

st.set_page_config(page_title="OpenAI SVG 平面圖框選測試", layout="wide")
st.title("OpenAI SVG 平面圖框選測試")
st.caption("圖面直接送入 OpenAI Vision，模型回傳與原圖同座標系統的 SVG Polygon。")

def render_pdf_first_page(pdf_bytes: bytes, dpi: int = 220) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not len(doc):
        doc.close()
        raise ValueError("PDF 沒有頁面。")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return image

def read_uploaded_file(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    if Path(uploaded_file.name).suffix.lower() == ".pdf":
        return render_pdf_first_page(content)
    return Image.open(io.BytesIO(content)).convert("RGB")

def resize_for_api(image: Image.Image, max_side: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)

def png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def overlay_html(image: Image.Image, svg: str) -> str:
    return f"""
    <div style="position:relative;width:100%;max-width:{image.width}px;margin:auto;line-height:0">
      <img src="data:image/png;base64,{png_b64(image)}" style="width:100%;height:auto;display:block">
      <div style="position:absolute;left:0;top:0;width:100%;height:100%">{svg}</div>
    </div>
    <style>svg{{width:100%;height:100%;display:block}}</style>
    """

uploaded = st.file_uploader("上傳 PDF、PNG 或 JPG", type=["pdf", "png", "jpg", "jpeg"])

with st.sidebar:
    model = st.text_input("OpenAI Vision 模型", value=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"))
    max_side = st.select_slider("送入 API 的最長邊", options=[1024, 1536, 2048, 2560], value=2048)

if uploaded:
    try:
        image = resize_for_api(read_uploaded_file(uploaded), max_side)
        st.write(f"實際送入 API：{image.width} × {image.height}")
        st.image(image, caption="送入 OpenAI 的完整圖面", use_container_width=True)

        if st.button("請 OpenAI 回傳房間 SVG", type="primary", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("尚未設定 OPENAI_API_KEY。")
                st.stop()
            with st.spinner("OpenAI 正在理解圖面並產生 SVG Polygon……"):
                st.session_state["svg_result"] = analyze_floorplan_as_svg(image, model)
                st.session_state["svg_image"] = image

        result = st.session_state.get("svg_result")
        result_image = st.session_state.get("svg_image")

        if result and result_image:
            st.subheader("SVG 直接疊加結果")
            components.html(overlay_html(result_image, result["svg"]), height=min(result_image.height + 30, 1000), scrolling=True)

            st.subheader("Python 解析 Polygon 後疊圖")
            st.image(draw_svg_polygons(result_image, result["rooms"]), use_container_width=True)

            st.dataframe([{
                "ID": room["id"],
                "名稱": room["name"],
                "角點數": len(room["points"]),
                "像素面積": round(room["area_pixels"], 2),
            } for room in result["rooms"]], use_container_width=True, hide_index=True)

            with st.expander("查看 OpenAI 原始 SVG"):
                st.code(result["svg"], language="xml")
            with st.expander("查看 Polygon JSON"):
                st.json(result["rooms"])

            st.download_button("下載 SVG", result["svg"], "floorplan_rooms.svg", "image/svg+xml", use_container_width=True)
            st.download_button("下載 Polygon JSON", json.dumps(result["rooms"], ensure_ascii=False, indent=2), "floorplan_rooms.json", "application/json", use_container_width=True)

    except Exception as exc:
        st.exception(exc)
