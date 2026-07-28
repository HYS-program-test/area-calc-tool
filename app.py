from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import numpy as np
import streamlit as st
from PIL import Image

from two_stage_room_analyzer import run_two_stage_analysis


st.set_page_config(
    page_title="平面圖兩階段 AI 房間辨識",
    layout="wide",
)

st.title("平面圖兩階段 AI 房間辨識")
st.caption(
    "第一階段只辨識房間與大致位置；第二階段逐房間局部放大並產生 Polygon。"
)


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

    return Image.open(io.BytesIO(content)).convert("RGB")


def crop_to_content(
    image: Image.Image,
    white_threshold: int = 246,
    padding_ratio: float = 0.025,
) -> Image.Image:
    """
    僅移除頁面外圍大片空白，不改變建築本體比例。
    """
    rgb = np.asarray(image.convert("RGB"))
    mask = np.any(rgb < white_threshold, axis=2)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return image

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1

    width, height = image.size
    padding = max(
        8,
        round(min(width, height) * padding_ratio),
    )

    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)

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

    crop_padding = st.slider(
        "局部房間裁切留白比例",
        min_value=0.05,
        max_value=0.35,
        value=0.18,
        step=0.01,
    )

    local_review_rounds = st.slider(
        "每個房間最多修正輪數",
        min_value=0,
        max_value=3,
        value=1,
    )


if uploaded_file is not None:
    try:
        page_image = read_uploaded_file(uploaded_file)
        building_image = crop_to_content(page_image)
        building_image = resize_working_image(
            building_image,
            max_side=max_side,
        )

        st.write(
            f"建築主體尺寸："
            f"{building_image.width} × {building_image.height}"
        )

        st.image(
            building_image,
            caption="建築主體",
            use_container_width=True,
        )

        if st.button(
            "執行兩階段 AI 分析",
            type="primary",
            use_container_width=True,
        ):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("尚未設定 OPENAI_API_KEY。")
                st.stop()

            with st.spinner(
                "AI 正在辨識房間、逐房間裁切並產生 Polygon……"
            ):
                result = run_two_stage_analysis(
                    building_image,
                    model=model,
                    crop_padding_ratio=crop_padding,
                    local_review_rounds=local_review_rounds,
                )

            st.session_state["two_stage_result"] = {
                "rooms": result["rooms"],
                "stage1_candidates": result[
                    "stage1_candidates"
                ],
                "logs": result["logs"],
                "image_size": result["image_size"],
            }

            st.session_state["stage1_overlay"] = result[
                "stage1_overlay"
            ]
            st.session_state["final_overlay"] = result[
                "final_overlay"
            ]
            st.session_state["local_crops"] = result[
                "local_crops"
            ]

        result_data = st.session_state.get(
            "two_stage_result"
        )
        stage1_overlay = st.session_state.get(
            "stage1_overlay"
        )
        final_overlay = st.session_state.get(
            "final_overlay"
        )
        local_crops = st.session_state.get(
            "local_crops",
            [],
        )

        if result_data:
            st.subheader("第一階段：房間候選位置")

            if stage1_overlay is not None:
                st.image(
                    stage1_overlay,
                    caption="AI 只辨識房間名稱與大致區域",
                    use_container_width=True,
                )

            st.subheader("第二階段：逐房間局部 Polygon")

            if final_overlay is not None:
                st.image(
                    final_overlay,
                    caption="局部分析結果合併回建築主體",
                    use_container_width=True,
                )

            st.subheader("局部裁切檢查")

            for item in local_crops:
                with st.expander(
                    f'{item["room_id"]}｜'
                    f'{item["room_name"]}'
                ):
                    st.image(
                        item["image"],
                        caption=(
                            f'局部裁切：'
                            f'{item["crop_box"]}'
                        ),
                        use_container_width=True,
                    )

            st.subheader("房間清單")

            st.dataframe(
                [
                    {
                        "ID": room["id"],
                        "名稱": room["name"],
                        "信心": room["confidence"],
                        "角點數": len(room["points"]),
                        "像素面積": round(
                            room["area_pixels"],
                            2,
                        ),
                    }
                    for room in result_data["rooms"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("第一階段候選 JSON"):
                st.json(
                    result_data["stage1_candidates"]
                )

            with st.expander("最終 Polygon JSON"):
                st.json(result_data["rooms"])

            with st.expander("處理紀錄"):
                st.json(result_data["logs"])

            st.download_button(
                "下載分析結果 JSON",
                data=json.dumps(
                    result_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name=(
                    "two_stage_floorplan_result.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
