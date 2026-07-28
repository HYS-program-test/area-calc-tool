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
st.caption(
    "AI 先產生候選 Polygon，再查看原圖與疊圖，逐輪回傳有限修正指令。"
)


def render_pdf_first_page(
    pdf_bytes: bytes,
    dpi: int = 250,
) -> Image.Image:
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    if len(document) == 0:
        document.close()
        raise ValueError("PDF 沒有頁面。")

    page = document[0]
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(pixmap.tobytes("png"))
    ).convert("RGB")

    document.close()
    return image


def read_uploaded_file(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return render_pdf_first_page(content)

    return Image.open(io.BytesIO(content)).convert("RGB")


def resize_for_api(
    image: Image.Image,
    max_side: int,
) -> Image.Image:
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
    "上傳 PDF、PNG 或 JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)

with st.sidebar:
    model = st.text_input(
        "OpenAI Vision 模型",
        value=os.getenv(
            "OPENAI_VISION_MODEL",
            "gpt-4.1",
        ),
    )

    max_side = st.select_slider(
        "送入 API 的最長邊",
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
        image = resize_for_api(
            read_uploaded_file(uploaded_file),
            max_side=max_side,
        )

        st.write(
            f"送入 API 的圖片尺寸："
            f"{image.width} × {image.height}"
        )

        st.image(
            image,
            caption="原始圖面",
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

            with st.spinner(
                "AI 正在產生候選框、查看疊圖並逐輪修正……"
            ):
                result = run_visual_review_loop(
                    image=image,
                    model=model,
                    max_rounds=max_rounds,
                )

            st.session_state["review_result"] = {
                "rooms": result["rooms"],
                "history": result["history"],
            }
            st.session_state["review_overlay"] = result[
                "final_overlay"
            ]

        result = st.session_state.get("review_result")
        overlay = st.session_state.get("review_overlay")

        if result and overlay:
            st.subheader("最終框選結果")
            st.image(
                overlay,
                caption="AI 多輪修正後的 Polygon",
                use_container_width=True,
            )

            st.subheader("空間清單")
            st.dataframe(
                [
                    {
                        "ID": room["id"],
                        "名稱": room["name"],
                        "角點數": len(room["points"]),
                        "信心": room["confidence"],
                        "像素面積": round(
                            room["area_pixels"],
                            2,
                        ),
                    }
                    for room in result["rooms"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("查看各輪修正紀錄"):
                st.json(result["history"])

            with st.expander("查看最終 Polygon JSON"):
                st.json(result["rooms"])

            st.download_button(
                "下載最終 JSON",
                data=json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="floorplan_review_result.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
