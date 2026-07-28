# OpenAI 平面圖原始判讀測試

這個版本只測試 OpenAI 對平面圖的原始理解能力。

## 流程

```text
上傳 PDF／圖片
→ PDF 第一頁轉圖片
→ 圖片直接送入 OpenAI Responses API
→ 顯示 output_text
→ 顯示完整 API Response JSON
```

本版不執行：

- 自動框線
- Polygon
- BBox
- OpenCV
- 裁切
- 座標轉換
- 面積計算
- 結果過濾
- Reviewer

目的在於確認模型到底能否正確理解：

- 圖面有哪些室內空間
- 各空間的位置與邊界
- 哪些是樓梯、設備、家具或室外空間
- 哪些區域容易誤判

## GitHub 最終檔案

只需上傳：

- `app.py`
- `requirements.txt`
- `README.md`

## Streamlit Cloud Secrets

```toml
OPENAI_API_KEY = "你的實際 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

`OPENAI_VISION_MODEL` 可不設定，程式頁面中也可以自行輸入模型名稱。

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 測試方式

1. 上傳原本的平面圖。
2. 使用預設 Prompt 執行一次。
3. 儲存：
   - 原始文字回覆
   - 完整 API Response JSON
4. 根據模型實際描述，再決定下一步適合：
   - 回傳語意定位
   - 回傳粗略區域
   - 結合牆角搜尋
   - 或停止使用純 Vision 座標方案
