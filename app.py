import io
import os
import fitz
import streamlit as st
from PIL import Image

from geometry import draw_result
from openai_test import analyze_with_openai
from pdf_utils import render_pdf_page

st.set_page_config(page_title="OpenAI 房間 Polygon 驗證", layout="wide")
st.title("OpenAI 房間 Polygon 真實驗證")

uploaded = st.file_uploader(
    "上傳平面圖 PDF 或建築圖片",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

model = st.text_input("模型", value="gpt-5.6")

if uploaded:
    file_name = uploaded.name.lower()
    file_bytes = uploaded.getvalue()
    image = None
    crop_metadata = None

    if file_name.endswith(".pdf"):
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                page_count = doc.page_count

            page_number = st.number_input(
                "選擇 PDF 頁碼",
                min_value=1,
                max_value=max(page_count, 1),
                value=1,
                step=1,
            )

            image, crop_metadata = render_pdf_page(
                file_bytes,
                page_index=int(page_number) - 1,
                dpi=220,
            )

            st.success(
                f"PDF 已讀取並自動裁切建築本體；"
                f"裁切方式：{crop_metadata['method']}。"
            )
            with st.expander("查看 PDF 裁切資訊"):
                st.json(crop_metadata)

        except Exception as exc:
            st.error(f"PDF 讀取或裁切失敗：{exc}")

    else:
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            st.info("圖片格式會直接進入辨識；PDF 才會執行向量式建築裁切。")
        except Exception as exc:
            st.error(f"圖片讀取失敗：{exc}")

    if image is not None:
        st.image(
            image,
            caption="送入 OpenAI 的建築圖片",
            use_container_width=True,
        )

        if st.button("呼叫 OpenAI 並框選房間", type="primary"):
            try:
                if "OPENAI_API_KEY" in st.secrets:
                    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

                with st.spinner("OpenAI 正在辨識房間 Polygon..."):
                    result = analyze_with_openai(image, model=model)
                    overlay, areas = draw_result(image, result)

                st.success(f"完成，共回傳 {len(result.rooms)} 個空間。")
                st.image(
                    overlay,
                    caption="Polygon 疊圖結果",
                    use_container_width=True,
                )

                st.subheader("結構化結果")
                st.json(result.model_dump())

                st.subheader("像素面積（尚未換算平方公尺）")
                st.dataframe(areas, use_container_width=True)

            except Exception as exc:
                st.exception(exc)
