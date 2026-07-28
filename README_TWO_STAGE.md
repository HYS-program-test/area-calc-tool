# 平面圖兩階段 OpenAI 測試版

## 架構

```text
建築主體
→ 第一階段：辨識房間名稱與 bbox
→ Python 逐房間局部裁切
→ 第二階段：局部放大後產生單一房間 Polygon
→ 可選擇局部 Reviewer
→ 換算回完整建築圖
```

## GitHub 更新

請同時上傳或覆蓋：

- `app.py`
- `two_stage_room_analyzer.py`
- `requirements.txt`

舊的 `visual_review_loop.py` 本版不會引用，可以保留，
但為避免混淆，建議移到 `archive/`。

## Streamlit Secrets

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

## 第一次測試建議

- 建築主體最長邊：2048
- 局部裁切留白比例：0.18
- 每個房間修正輪數：0

先看第二階段本身的結果，再將修正輪數調為 1。

## API 呼叫數量

假設第一階段辨識出 6 個房間：

- Stage 1：1 次
- Stage 2：6 次
- Reviewer 設為 0：共 7 次
- Reviewer 設為 1：最多共 13 次

請注意 API 使用費。
