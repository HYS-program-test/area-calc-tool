from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import numpy as np
import streamlit as st
from PIL import Image

from visual_review_loop import run_visual_review_loop


st.set_page_config(
    page_title="平面圖 AI 視覺修正迴圈",
    layout="wide",
)

st.title("平面圖 AI 視覺修正迴圈")
st.caption("保留建築主體，只修正比例與座標系統。")


def render_pdf_first_page(
    pdf_bytes: bytes,
    dpi: int = 220,
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

    return Image.open(
        io.BytesIO(content)
    ).convert("RGB")


def crop_to_building_content(
    image: Image.Image,
    white_threshold: int = 246,
    padding_ratio: float = 0.025,
) -> Image.Image:
    """
    只移除頁面四周大面積空白，不加入上下左右界控制，
    也不改變建築本體比例。
    """
    rgb = np.asarray(image.convert("RGB"))
    content_mask = np.any(rgb < white_threshold, axis=2)
    ys, xs = np.where(content_mask)

    if len(xs) == 0 or len(ys) == 0:
        return image

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1

    width, height = image.size
    padding = max(8, round(min(width, height) * padding_ratio))

    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)

    # 如果偵測結果幾乎等於整張圖，就保留原圖，避免錯誤裁切。
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width >= width * 0.97 and crop_height >= height * 0.97:
        return image

    return image.crop((x0, y0, x1, y1))


def resize_working_image(
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
        "建築主體最長邊",
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
        page_image = read_uploaded_file(uploaded_file)

        # 只移除四周空白；介面、API、Polygon 全部使用同一張 building_image。
        building_image = crop_to_building_content(page_image)
        building_image = resize_working_image(
            building_image,
            max_side=max_side,
        )

        st.write(
            f"建築主體圖片尺寸："
            f"{building_image.width} × {building_image.height}"
        )

        st.image(
            building_image,
            caption="建築物主體",
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
                "AI 正在使用標準化座標進行框選與修正……"
            ):
                # 使用位置參數，避免 original_image 關鍵字名稱不一致。
                result = run_visual_review_loop(
                    building_image,
                    model,
                    max_rounds,
                )

            st.session_state["review_result"] = {
                "rooms": result["rooms"],
                "history": result["history"],
                "image_size": result.get(
                    "image_size",
                    [
                        building_image.width,
                        building_image.height,
                    ],
                ),
            }

            st.session_state["review_grid"] = result.get(
                "gridded_image"
            )

            st.session_state["review_overlay"] = result[
                "final_overlay"
            ]

        result_data = st.session_state.get("review_result")
        gridded_image = st.session_state.get("review_grid")
        overlay = st.session_state.get("review_overlay")

        if result_data and overlay:
            if gridded_image is not None:
                st.subheader("AI 座標定位圖")
                st.image(
                    gridded_image,
                    caption="建築主體＋0～1000 標準座標",
                    use_container_width=True,
                )

            st.subheader("最終 Polygon")
            st.image(
                overlay,
                caption="建築主體框選結果",
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
                            room.get("area_pixels", 0),
                            2,
                        ),
                    }
                    for room in result_data["rooms"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("查看最終 Polygon JSON"):
                st.json(result_data["rooms"])

            with st.expander("查看各輪修正紀錄"):
                st.json(result_data["history"])

            st.download_button(
                "下載結果 JSON",
                data=json.dumps(
                    result_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="floorplan_review_result.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
