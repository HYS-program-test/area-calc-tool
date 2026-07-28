from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

from visual_review_loop import run_visual_review_loop


st.set_page_config(
    page_title="平面圖 AI 視覺修正迴圈",
    layout="wide",
)

st.title("平面圖 AI 視覺修正迴圈")
st.caption("保留既有建築主體，不增加上下左右裁切控制；只修正比例與座標系統。")


def render_pdf_first_page(pdf_bytes: bytes, dpi: int = 220) -> Image.Image:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(document) == 0:
        document.close()
        raise ValueError("PDF 沒有頁面。")

    page = document[0]
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    document.close()
    return image


def read_uploaded_file(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return render_pdf_first_page(content)

    return Image.open(io.BytesIO(content)).convert("RGB")


def resize_working_image(image: Image.Image, max_side: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_side:
        return image

    scale = max_side / longest
    return image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


uploaded_file = st.file_uploader(
    "上傳已完成建築主體裁切的 PDF、PNG 或 JPG",
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
        original = resize_working_image(
            read_uploaded_file(uploaded_file),
            max_side=max_side,
        )

        st.write(f"工作圖片尺寸：{original.width} × {original.height}")
        st.image(
            original,
            caption="目前既有的建築主體圖",
            use_container_width=True,
        )

        if st.button(
            "執行 AI 視覺修正迴圈",
            type="primary",
            use_container_width=True,
        ):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("尚未設定 OPENAI_API_KEY。")
                st.stop()

            with st.spinner("AI 正在使用標準化座標進行框選與修正……"):
                result = run_visual_review_loop(
                    original_image=original,
                    model=model,
                    max_rounds=max_rounds,
                )

            st.session_state["review_result_v3"] = {
                "rooms": result["rooms"],
                "history": result["history"],
                "image_size": result["image_size"],
            }
            st.session_state["grid_v3"] = result["gridded_image"]
            st.session_state["overlay_v3"] = result["final_overlay"]

        result_data = st.session_state.get("review_result_v3")
        gridded_image = st.session_state.get("grid_v3")
        overlay = st.session_state.get("overlay_v3")

        if result_data and gridded_image and overlay:
            st.subheader("AI 實際看到的 0～1000 座標格線")
            st.image(gridded_image, use_container_width=True)

            st.subheader("最終 Polygon")
            st.image(overlay, use_container_width=True)

            st.subheader("空間清單")
            st.dataframe(
                [{
                    "ID": room["id"],
                    "名稱": room["name"],
                    "角點數": len(room["points"]),
                    "信心": room["confidence"],
                    "像素面積": round(room["area_pixels"], 2),
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
                file_name="floorplan_review_result_v3.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
