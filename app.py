from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

from ai_image_annotator import annotate_floorplan
from red_line_extractor import extract_red_polygons, draw_polygons


st.set_page_config(page_title="AI 平面圖框選測試", layout="wide")
st.title("AI 平面圖框選測試")
st.caption("將完整圖面交給 OpenAI 圖像編輯模型，要求它直接在圖面上畫紅色空間框。")


def render_pdf_first_page(pdf_bytes: bytes, dpi: int = 220) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        doc.close()
        raise ValueError("PDF 沒有頁面。")

    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return image


def read_uploaded_file(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return render_pdf_first_page(content)

    return Image.open(io.BytesIO(content)).convert("RGB")


def resize_for_api(image: Image.Image, max_side: int = 1536) -> Image.Image:
    width, height = image.size
    longest = max(width, height)

    if longest <= max_side:
        return image

    ratio = max_side / longest
    return image.resize(
        (max(1, round(width * ratio)), max(1, round(height * ratio))),
        Image.Resampling.LANCZOS,
    )


uploaded_file = st.file_uploader(
    "上傳 PDF、PNG 或 JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)

model = st.text_input(
    "OpenAI 圖像模型",
    value=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
)

if uploaded_file is not None:
    source_image = resize_for_api(read_uploaded_file(uploaded_file))

    st.image(source_image, caption="送入 OpenAI 的原始圖面", use_container_width=True)

    if st.button("交給 OpenAI 畫出室內空間框", type="primary", use_container_width=True):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("尚未設定 OPENAI_API_KEY。")
            st.stop()

        with st.spinner("OpenAI 正在理解圖面並直接畫出紅色空間框……"):
            annotated_image = annotate_floorplan(
                source_image,
                model=model,
            )

        with st.spinner("正在從 AI 回傳圖中擷取紅色框線……"):
            polygons = extract_red_polygons(annotated_image)
            verified_overlay = draw_polygons(source_image, polygons)

        st.session_state["annotated_image"] = annotated_image
        st.session_state["verified_overlay"] = verified_overlay
        st.session_state["polygons"] = polygons

    annotated_image = st.session_state.get("annotated_image")
    verified_overlay = st.session_state.get("verified_overlay")
    polygons = st.session_state.get("polygons")

    if annotated_image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("OpenAI 回傳圖")
            st.image(
                annotated_image,
                caption="AI 直接在圖上畫出的紅色框",
                use_container_width=True,
            )

        with col2:
            st.subheader("紅線擷取結果")
            st.image(
                verified_overlay,
                caption="Python 從紅線轉回 Polygon 後疊回原圖",
                use_container_width=True,
            )

        st.subheader("Polygon JSON")
        st.json(polygons)

        json_text = json.dumps(polygons, ensure_ascii=False, indent=2)
        st.download_button(
            "下載 Polygon JSON",
            data=json_text,
            file_name="floorplan_polygons.json",
            mime="application/json",
            use_container_width=True,
        )
