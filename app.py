import os
import streamlit as st
from PIL import Image
from geometry import draw_result
from openai_test import analyze_with_openai

st.set_page_config(page_title="OpenAI 房間 Polygon 驗證", layout="wide")
st.title("OpenAI 房間 Polygon 真實驗證")

uploaded = st.file_uploader("上傳裁切後的建築圖片", type=["png", "jpg", "jpeg"])
model = st.text_input("模型", value="gpt-5.6")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="輸入圖片", use_container_width=True)

    if st.button("呼叫 OpenAI 並框選房間", type="primary"):
        try:
            if "OPENAI_API_KEY" in st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

            with st.spinner("OpenAI 正在辨識房間 Polygon..."):
                result = analyze_with_openai(image, model=model)
                overlay, areas = draw_result(image, result)

            st.success(f"完成，共回傳 {len(result.rooms)} 個空間。")
            st.image(overlay, caption="Polygon 疊圖結果", use_container_width=True)
            st.subheader("結構化結果")
            st.json(result.model_dump())
            st.subheader("像素面積（尚未換算平方公尺）")
            st.dataframe(areas, use_container_width=True)
        except Exception as exc:
            st.exception(exc)
