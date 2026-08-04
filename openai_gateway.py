import base64
import io
import os
from PIL import Image
from openai import OpenAI
from schemas import FloorplanResult

SYSTEM_PROMPT = """
你是建築平面圖空間邊界分析器，任務是把一張樓層平面圖轉換成每個獨立空間的多邊形座標。

# 判斷哪些算「空間」
- 只辨識可供人員使用、由牆體圍成的室內獨立空間。
- 不要把樓梯、電梯井、管道間、牆體本身、柱子、家具、衣櫃，或圖面外的空白當成房間。
- 走廊、陽台、廁所、儲藏室這類「有牆圍起來、但不是臥室/客廳」的空間，只要是由實牆圍成、
  可計算面積的獨立空間，一樣要當成一個空間輸出（不要因為它不是「主要用途房間」就略過）。
- 兩個空間之間如果只是開放式門洞相連（沒有實際隔間牆），仍然算兩個獨立空間，
  不要因為門是開著的就合併成一個多邊形。

# 如何提高準確度
- 平面圖上如果印有房間名稱文字（例如「臥室」「客廳」「主臥」），把該文字原樣填進
  label_on_plan 欄位，並以這個文字標籤的位置為準去找對應的空間邊界——文字標籤是
  比單純目測牆線更可靠的線索，優先信任它。
- 每個空間先用一句話寫下你怎麼判斷這個空間邊界的（wall_trace_notes），
  再給座標；不規則形狀（L 形、凹凸形）的空間要用足夠多的頂點確實沿著牆角描繪，
  不要為了省事只給 4 個頂點把不規則空間硬套成矩形。
- 全部空間辨識完後，在 room_count_check 欄位用一句話自我檢查：這個數量看起來合理嗎？
  平面圖上有沒有看起來還有牆圍起來的區域、但你沒有輸出成獨立空間的？如果有懷疑遺漏，
  在這裡明講。

# 座標規則
座標使用 0 到 1000，左上角=(0,0)，右下角=(1000,1000)。
多邊形不可超出圖片，不可互相大幅重疊，也不可跨越實牆。
points 依實際牆角數量給頂點（不是固定 4 點），依順時針或逆時針排列。
"""

USER_PROMPT = """
分析這張已自動裁切、只保留建築本體的樓層平面圖。
輸出所有由實牆圍成、可以計算室內面積的獨立空間多邊形，包含走廊、陽台、廁所等
非主要用途空間。如果平面圖上印有房間名稱文字，請對照文字標籤來確認空間邊界。
id 從 1 開始；name 使用區域1、區域2；color 依序使用
#ef4444、#f97316、#f59e0b、#22c55e、#3b82f6、#a855f7，
超過 6 個空間就從頭重複使用這組顏色。
"""

def _data_url(image: Image.Image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def analyze_floorplan(image: Image.Image, model: str = "gpt-5.6") -> FloorplanResult:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("找不到 OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    response = client.responses.parse(
        model=model,
        input=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[
                {"type":"input_text","text":USER_PROMPT},
                {"type":"input_image","image_url":_data_url(image),"detail":"high"},
            ]},
        ],
        text_format=FloorplanResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI 未回傳可解析結果")
    return response.output_parsed
