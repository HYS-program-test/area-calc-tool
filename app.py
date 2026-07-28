from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

from src.area_calc.config import AppConfig
from src.area_calc.pipeline import analyze_floorplan


st.set_page_config(
    page_title="平面圖空間辨識",
    layout="wide",
)

st.title("平面圖空間辨識")
st.caption(
    "固定架構版：第一階段辨識房間候選，第二階段逐房間局部框選。"
)


def render_pdf_first_page(
    pdf_bytes: bytes,
    dpi: int,
) -> Image.Image:
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        if len(document) == 0:
            raise ValueError("PDF 沒有頁面。")

        page = document[0]
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            alpha=False,
        )

        return Image.open(
            io.BytesIO(pixmap.tobytes("png"))
        ).convert("RGB")
    finally:
        document.close()


def read_uploaded_file(
    uploaded_file,
    dpi: int,
) -> Image.Image:
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return render_pdf_first_page(raw, dpi)

    return Image.open(
        io.BytesIO(raw)
    ).convert("RGB")


config = AppConfig.from_environment()

with st.sidebar:
    model = st.text_input(
        "OpenAI 視覺模型",
        value=config.model,
    )

    pdf_dpi = st.select_slider(
        "PDF 解析度",
        options=[180, 220, 250, 300],
        value=config.pdf_dpi,
    )

    max_side = st.select_slider(
        "送入分析的最長邊",
        options=[1536, 2048, 2560],
        value=config.max_side,
    )

    stage2_padding = st.slider(
        "局部裁切留白",
        min_value=0.05,
        max_value=0.30,
        value=config.stage2_padding,
        step=0.01,
    )

    review_rounds = st.slider(
        "每個空間修正輪數",
        min_value=0,
        max_value=2,
        value=config.review_rounds,
    )

    mock_mode = st.checkbox(
        "沙盒模擬模式",
        value=False,
        help="不呼叫 OpenAI，只驗證介面、座標與合併流程。",
    )


uploaded_file = st.file_uploader(
    "上傳 PDF、PNG 或 JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        image = read_uploaded_file(
            uploaded_file,
            pdf_dpi,
        )

        st.image(
            image,
            caption="上傳圖面",
            use_container_width=True,
        )

        if st.button(
            "開始分析",
            type="primary",
            use_container_width=True,
        ):
            if (
                not mock_mode
                and not os.getenv("OPENAI_API_KEY")
            ):
                st.error(
                    "尚未設定 OPENAI_API_KEY。"
                )
                st.stop()

            run_config = AppConfig(
                model=model,
                pdf_dpi=pdf_dpi,
                max_side=max_side,
                stage2_padding=stage2_padding,
                review_rounds=review_rounds,
            )

            with st.spinner(
                "正在進行兩階段分析……"
            ):
                result = analyze_floorplan(
                    image=image,
                    config=run_config,
                    mock_mode=mock_mode,
                )

            st.session_state[
                "area_calc_result"
            ] = result.serializable()

            st.session_state[
                "area_calc_stage1"
            ] = result.stage1_overlay

            st.session_state[
                "area_calc_final"
            ] = result.final_overlay

            st.session_state[
                "area_calc_local"
            ] = result.local_previews

        result_data = st.session_state.get(
            "area_calc_result"
        )

        if result_data:
            st.subheader("第一階段候選")
            st.image(
                st.session_state[
                    "area_calc_stage1"
                ],
                use_container_width=True,
            )

            st.subheader("第二階段合併結果")
            st.image(
                st.session_state[
                    "area_calc_final"
                ],
                use_container_width=True,
            )

            st.subheader("局部分析圖")
            for preview in st.session_state.get(
                "area_calc_local",
                [],
            ):
                with st.expander(
                    f'{preview["id"]}｜'
                    f'{preview["name"]}'
                ):
                    st.image(
                        preview["image"],
                        use_container_width=True,
                    )

            st.subheader("空間資料")
            st.dataframe(
                result_data["rooms"],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("完整分析紀錄"):
                st.json(result_data)

            st.download_button(
                "下載 JSON",
                data=json.dumps(
                    result_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="area_calc_result.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
