# Area Calc Clean v1

這是一個重新整理後的乾淨專案。

## 固定架構

```text
app.py
└── src/area_calc/
    ├── config.py
    ├── schemas.py
    ├── prompts.py
    ├── geometry.py
    ├── image_ops.py
    ├── openai_gateway.py
    ├── mock_gateway.py
    └── pipeline.py
```

後續調整 AI 判讀時，主要修改：

- `prompts.py`
- `openai_gateway.py`
- `pipeline.py`

不再建立多個 analyzer、svg、red-line、review-loop 實驗檔。

## 第一次部署

1. 建立全新的 GitHub Repository。
2. 將壓縮檔全部內容上傳到 Repository 根目錄。
3. Streamlit 主程式指定 `app.py`。
4. 在 Streamlit Secrets 設定：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

5. 先勾選「沙盒模擬模式」測試介面與座標。
6. 模擬模式正常後取消勾選，才呼叫 OpenAI。

## 建議 OpenAI 測試參數

- 最長邊：2048
- 局部裁切留白：0.18
- Reviewer：0

確認 Stage 2 原始結果後，再把 Reviewer 改為 1。
