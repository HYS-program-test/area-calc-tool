from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import fitz
import streamlit as st
from openai import OpenAI
from PIL import Image


st.set_page_config(
    page_title="OpenAI 平面圖原始判讀測試",
    layout="wide",
)

st.title("OpenAI 平面圖原始判讀測試")
st.caption(
    "這個版本不畫框、不轉座標、不過濾結果；"
    "只把圖面交給 OpenAI，完整顯示模型原始回覆。"
)


DEFAULT_PROMPT = """
請直接閱讀這張建築平面圖。

本次不要輸出座標、Polygon、BBox，也不要在圖片上畫線。
請只做空間理解與文字說明。

請依下列順序回覆：

1. 你認為主要建築平面圖位於圖片的哪個區域。
2. 逐一列出你辨識到的主要室內使用空間。
3. 對每一個空間說明：
   - 推測用途
   - 位於圖面的哪個位置
   - 由哪些牆、門洞或相鄰空間界定
   - 是否為完整封閉空間、開放式空間，或無法確定
4. 列出你認為不應計入室內使用面積的區域，例如：
   - 樓梯
   - 電梯設備
   - 家具或櫃體
   - 陽台、庭院、車道
   - 尺寸線、文字與圖面標註
5. 說明圖面中哪些區域最容易被誤判。
6. 最後用一段話總結：若下一步要框出各室內空間，你會優先框哪些區域。

請依你真正看見的圖面回答。
看不清楚或無法確定時，請明確說明，不要自行補充。
"""


def render_pdf_first_page(
    pdf_bytes: bytes,
    dpi: int = 220,
) -> Image.Image:
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
) -> tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)

    if longest <= max_side:
        return image, 1.0

    scale = max_side / longest
    resized = image.resize(
        (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def image_to_data_url(
    image: Image.Image,
    quality: int = 95,
) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()

    if hasattr(response, "to_dict"):
        return response.to_dict()

    return {"repr": repr(response)}


def ask_openai(
    image: Image.Image,
    prompt: str,
    model: str,
    detail: str,
) -> tuple[str, dict[str, Any]]:
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            prompt
                            + f"\n\n輸入圖片尺寸："
                            + f"{image.width} × {image.height} 像素。"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image),
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    return response.output_text, response_to_dict(response)


uploaded_file = st.file_uploader(
    "上傳 PDF、PNG 或 JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)

with st.sidebar:
    st.header("API 設定")

    model = st.text_input(
        "模型名稱",
        value=os.getenv(
            "OPENAI_VISION_MODEL",
            "gpt-4.1",
        ),
    )

    max_side = st.select_slider(
        "送入 API 的最長邊",
        options=[1024, 1536, 2048, 2560],
        value=2048,
    )

    detail = st.selectbox(
        "圖片 detail",
        options=["high", "auto", "low"],
        index=0,
    )

    prompt = st.text_area(
        "測試 Prompt",
        value=DEFAULT_PROMPT.strip(),
        height=520,
    )


if uploaded_file is not None:
    try:
        original_image = read_uploaded_file(uploaded_file)
        api_image, scale = resize_for_api(
            original_image,
            max_side=max_side,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "原始圖片尺寸",
                f"{original_image.width} × {original_image.height}",
            )

        with col2:
            st.metric(
                "API 圖片尺寸",
                f"{api_image.width} × {api_image.height}",
            )

        st.image(
            api_image,
            caption="實際送入 OpenAI API 的圖片",
            use_container_width=True,
        )

        if st.button(
            "將圖面交給 OpenAI 原始判讀",
            type="primary",
            use_container_width=True,
        ):
            if not os.getenv("OPENAI_API_KEY"):
                st.error(
                    "尚未設定 OPENAI_API_KEY。"
                    "請在 Streamlit Cloud Secrets 中設定。"
                )
                st.stop()

            with st.spinner("OpenAI 正在閱讀圖面……"):
                output_text, raw_response = ask_openai(
                    api_image,
                    prompt=prompt,
                    model=model,
                    detail=detail,
                )

            st.session_state["output_text"] = output_text
            st.session_state["raw_response"] = raw_response
            st.session_state["used_prompt"] = prompt
            st.session_state["used_model"] = model

        output_text = st.session_state.get("output_text")
        raw_response = st.session_state.get("raw_response")

        if output_text is not None:
            st.subheader("OpenAI 原始文字回覆")
            st.markdown(output_text)

            st.download_button(
                "下載原始文字回覆",
                data=output_text,
                file_name="openai_floorplan_raw_response.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if raw_response is not None:
            with st.expander("查看完整 API Response"):
                st.json(raw_response)

            raw_json = json.dumps(
                raw_response,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            st.download_button(
                "下載完整 API Response JSON",
                data=raw_json,
                file_name="openai_floorplan_full_response.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
