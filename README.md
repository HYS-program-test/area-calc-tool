# 平面圖 AI 視覺修正迴圈 v2

本版修正重點：

1. 不再要求模型輸出圖片像素座標。
2. 所有 AI 座標改為 `0～1000` 標準化座標。
3. AI 實際看到的裁切圖加入座標格線。
4. 使用者先調整建築主體裁切範圍。
5. Polygon 在裁切圖上產生，再換算回原圖。
6. Reviewer 仍只能回傳有限修正命令。

## 流程

```text
上傳 PDF / 圖片
→ 調整建築裁切範圍
→ 建立 0～1000 座標格線圖
→ AI 產生初始 Polygon
→ Python 畫出角點
→ AI Reviewer 檢查
→ move/add/delete point
→ 最多重複 1～5 輪
→ 換算回原圖座標
```

## Streamlit Secrets

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 建議第一次測試

- 工作圖片最長邊：2048
- 最多修正輪數：3
- 裁切紅框只保留建築本體
