import base64
import io
import os
from PIL import Image
from openai import OpenAI
from schemas import FloorplanResult

SYSTEM_PROMPT = """
你是建築平面圖空間邊界分析器。
只辨識可供人員使用、由牆體圍成的室內獨立空間。
不要把樓梯、電梯井、管道間、牆體、柱子、家具、衣櫃或圖面外空白當成房間。
每個空間以多邊形表示，點必須沿著空間內側牆面排列。
座標使用 0 到 1000，左上角=(0,0)，右下角=(1000,1000)。
多邊形不可超出圖片，不可互相大幅重疊，也不可跨越實牆。
points 按順時針或逆時針排列，至少 4 點。
無法可靠判斷的空間不要輸出。
"""

USER_PROMPT = """
分析這張已自動裁切、只保留建築主體的樓層平面圖。
輸出所有可以計算室內面積的獨立空間多邊形。
優先辨識臥室、客餐廳、衛浴、更衣室與其他封閉房間。
"""

def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def analyze_with_openai(image: Image.Image, model: str) -> FloorplanResult:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("找不到 OPENAI_API_KEY。請在 Streamlit Secrets 設定。")

    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image),
                        "detail": "high",
                    },
                ],
            },
        ],
        text_format=FloorplanResult,
    )

    if response.output_parsed is None:
        raise RuntimeError("OpenAI 沒有回傳可解析的 Polygon 結果。")

    return response.output_parsed
