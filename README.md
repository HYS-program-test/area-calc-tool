# AI 直接畫框測試版

此版本的目的，是測試以下流程：

```text
上傳平面圖
→ 將完整圖面交給 OpenAI Images Edit API
→ 要求 AI 保留原圖，只增加純紅色封閉空間框
→ Python 擷取紅色線
→ 轉成 Polygon
→ 將 Polygon 疊回原始圖面
```

## 檔案

- `app.py`
- `ai_image_annotator.py`
- `red_line_extractor.py`
- `requirements.txt`

## Streamlit Cloud Secrets

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_IMAGE_MODEL = "gpt-image-1"
```

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 重要限制

這是實驗流程。圖像生成／編輯模型可能重新繪製原圖中的線條、文字或尺寸，
即使 Prompt 要求保持不變，也不能保證像素完全一致。

因此頁面同時顯示：

1. OpenAI 回傳的編輯圖。
2. Python 從紅線擷取後，重新疊回原始圖的結果。

應以第二張圖判斷紅線擷取是否可用。
