from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image, ImageDraw

from visual_review_loop import run_visual_review_loop


st.set_page_config(page_title="平面圖 AI 視覺修正迴圈 v2", layout="wide")
st.title("平面圖 AI 視覺修正迴圈 v2")
st.caption("建築裁切＋0～1000 標準座標＋格線＋多輪 Reviewer")


def render_pdf_first_page(pdf_bytes: bytes, dpi: int = 220) -> Image.Image:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(document) == 0:
        document.close()
        raise ValueError("PDF 沒有頁面。")

    page = document[0]
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    document.close()
    return image


def read_uploaded_file(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        return render_pdf_first_page(content)
    return Image.open(io.BytesIO(content)).convert("RGB")


def resize_for_working_image(image: Image.Image, max_side: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )


def preview_crop_box(image: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    left, top, right, bottom = crop_box
    draw.rectangle((left, top, right, bottom), outline=(255, 0, 0, 255), width=5)
    draw.rectangle((left, top, right, bottom), fill=(255, 0, 0, 18))
    return result


uploaded_file = st.file_uploader(
    "上傳 PDF、PNG 或 JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)

with st.sidebar:
    model = st.text_input(
        "OpenAI Vision 模型",
        value=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"),
    )
    max_side = st.select_slider(
        "工作圖片最長邊",
        options=[1536, 2048, 2560],
        value=2048,
    )
    max_rounds = st.slider(
        "最多修正輪數",
        min_value=1,
        max_value=5,
        value=3,
    )

if uploaded_file is not None:
    try:
        original = resize_for_working_image(
            read_uploaded_file(uploaded_file),
            max_side=max_side,
        )
        width, height = original.size

        st.write(f"工作圖片尺寸：{width} × {height}")

        st.subheader("步驟一：設定建築主體裁切範圍")
        st.caption("請把紅框調整到只保留建築主體，盡量排除基地線、道路、大面積空白與尺寸標註。")

        col1, col2 = st.columns(2)
        with col1:
            left_pct = st.slider("左界 (%)", 0, 90, 5)
            top_pct = st.slider("上界 (%)", 0, 90, 5)
        with col2:
            right_pct = st.slider("右界 (%)", 10, 100, 95)
            bottom_pct = st.slider("下界 (%)", 10, 100, 95)

        left = round(width * left_pct / 100)
        top = round(height * top_pct / 100)
        right = round(width * right_pct / 100)
        bottom = round(height * bottom_pct / 100)

        if right <= left or bottom <= top:
            st.error("裁切範圍無效：右界必須大於左界，下界必須大於上界。")
            st.stop()

        crop_box = (left, top, right, bottom)

        crop_preview = preview_crop_box(original, crop_box)
        st.image(crop_preview, caption="紅框為送入 AI 的建築裁切範圍", use_container_width=True)

        crop_image = original.crop(crop_box)
        st.image(crop_image, caption="裁切後建築主體", use_container_width=True)

        if st.button("執行 AI 視覺修正迴圈", type="primary", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("尚未設定 OPENAI_API_KEY。")
                st.stop()

            with st.spinner("AI 正在產生候選框、查看格線疊圖並逐輪修正……"):
                result = run_visual_review_loop(
                    original_image=original,
                    crop_box=crop_box,
                    model=model,
                    max_rounds=max_rounds,
                )

            st.session_state["review_result_v2"] = {
                "rooms": result["rooms"],
                "history": result["history"],
                "crop_box": result["crop_box"],
                "crop_size": result["crop_size"],
            }
            st.session_state["gridded_crop_v2"] = result["gridded_crop"]
            st.session_state["crop_overlay_v2"] = result["final_crop_overlay"]
            st.session_state["original_overlay_v2"] = result["final_original_overlay"]

        result_data = st.session_state.get("review_result_v2")
        gridded_crop = st.session_state.get("gridded_crop_v2")
        crop_overlay = st.session_state.get("crop_overlay_v2")
        original_overlay = st.session_state.get("original_overlay_v2")

        if result_data and gridded_crop and crop_overlay and original_overlay:
            st.subheader("AI 實際看到的座標格線圖")
            st.image(gridded_crop, use_container_width=True)

            st.subheader("裁切區域的最終 Polygon")
            st.image(crop_overlay, use_container_width=True)

            st.subheader("換算回原圖後的最終 Polygon")
            st.image(original_overlay, use_container_width=True)

            st.subheader("空間清單")
            st.dataframe(
                [{
                    "ID": room["id"],
                    "名稱": room["name"],
                    "角點數": len(room["points"]),
                    "信心": room["confidence"],
                    "裁切圖像素面積": round(room["area_pixels_on_crop"], 2),
                } for room in result_data["rooms"]],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("查看最終 Polygon JSON"):
                st.json(result_data["rooms"])

            with st.expander("查看各輪修正紀錄"):
                st.json(result_data["history"])

            st.download_button(
                "下載結果 JSON",
                data=json.dumps(result_data, ensure_ascii=False, indent=2),
                file_name="floorplan_review_result_v2.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
