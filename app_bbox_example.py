from __future__ import annotations

import io
import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

from room_bbox_detector import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DetectionOptions,
    run_same_flow_as_manual,
)

# 沿用你目前可正常顯示底圖及框線的編輯器。
from floorplan_editor import floorplan_editor


st.set_page_config(page_title="平面圖空間框選", layout="wide")
st.title("平面圖空間框選")


def render_pdf_first_page(pdf_bytes: bytes, dpi: int = 300) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        raise ValueError("PDF 沒有頁面。")

    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return image


def read_uploaded_image(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return render_pdf_first_page(content, dpi=300)

    return Image.open(io.BytesIO(content)).convert("RGB")


uploaded_file = st.file_uploader(
    "上傳 PDF 或圖片",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        original_image = read_uploaded_image(uploaded_file)

        model = st.text_input(
            "Vision 模型",
            value=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"),
        )

        run_detection = st.button(
            "依人工流程自動框選",
            type="primary",
            use_container_width=True,
        )

        if run_detection:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("尚未設定 OPENAI_API_KEY。")
                st.stop()

            with st.spinner(
                "正在找建築主體、裁切、放大並框選室內空間……"
            ):
                result = run_same_flow_as_manual(
                    original_image,
                    DetectionOptions(model=model),
                )

                st.session_state["floorplan_result"] = {
                    "crop_result": result["crop_result"],
                    "coordinate_info": result["coordinate_info"],
                    "room_detection": result["room_detection"],
                    "polygons": result["polygons"],
                }
                st.session_state["floorplan_canvas_image"] = result[
                    "canvas_image"
                ]
                st.session_state["floorplan_cropped_image"] = result[
                    "cropped_image"
                ]

        result_data = st.session_state.get("floorplan_result")
        canvas_image = st.session_state.get("floorplan_canvas_image")
        cropped_image = st.session_state.get("floorplan_cropped_image")

        if result_data and canvas_image:
            st.subheader("處理結果")

            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    original_image,
                    caption="原始 PDF／圖片",
                    use_container_width=True,
                )
            with col2:
                st.image(
                    cropped_image,
                    caption="OpenAI 找到建築主體後的裁切圖",
                    use_container_width=True,
                )

            st.caption(
                f"裁切後固定畫布：{CANVAS_WIDTH} × {CANVAS_HEIGHT}"
            )

            polygons = result_data["polygons"]

            # 若你的 floorplan_editor 參數名稱不同，
            # 只需調整下面這一段。
            edited_polygons = floorplan_editor(
                image=canvas_image,
                rooms=polygons,
                width=CANVAS_WIDTH,
                height=CANVAS_HEIGHT,
                key="floorplan_room_editor",
            )

            if edited_polygons:
                # 編輯後 points 是 Canvas 座標。
                # 若要匯出回原圖，需使用 coordinate_info 再轉換。
                result_data["polygons"] = edited_polygons
                st.session_state["floorplan_result"] = result_data

            with st.expander("查看裁切座標"):
                st.json(result_data["crop_result"])

            with st.expander("查看三層座標資訊"):
                st.json(result_data["coordinate_info"])

            with st.expander("查看房間與 Polygon 座標"):
                st.json(result_data["polygons"])

    except Exception as exc:
        st.exception(exc)
