# OpenAI SVG 平面圖框選測試

流程：

```text
上傳 PDF／圖片
→ 圖面直接送入 OpenAI Vision
→ OpenAI 只回傳 SVG
→ 每個空間是一個 polygon
→ SVG 疊加原圖
→ Python 解析 polygon 並重新畫回原圖
```

## 檔案

- `app.py`
- `openai_svg_analyzer.py`
- `requirements.txt`
- `README.md`

## Streamlit Secrets

```toml
OPENAI_API_KEY = "你的實際 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

SVG 是另一種座標輸出格式，不保證空間定位必然比 JSON Polygon 準確。本版用於驗證模型是否較能以 SVG 表達空間邊界。
