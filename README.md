# OpenAI Direct Floorplan Analyzer

這版只聚焦在一條路：

```text
上傳 PDF／圖片
→ PDF 第一頁轉成高解析度圖片
→ 完整圖片直接交給 OpenAI Vision API
→ OpenAI 回傳空間 Polygon 與比例資訊
→ Python 畫框並計算面積
```

沒有使用：

- OpenCV 牆線偵測
- 自動裁切 Agent
- BBox 過濾流程
- 多次空間盤點
- Reviewer Agent
- 固定房間座標

## 檔案

- `app_openai_direct.py`
- `openai_floorplan_analyzer.py`
- `requirements.txt`

## 安裝

```bash
pip install -r requirements.txt
```

## Streamlit Cloud Secrets

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

## 執行

```bash
streamlit run app_openai_direct.py
```

## 座標設計

送入 OpenAI 的圖片與頁面顯示底圖使用同一個尺寸，因此：

- OpenAI 回傳的 Polygon 是圖片實際像素座標。
- 不需要 crop 座標轉換。
- 不需要 0~1000 正規化座標。
- 不需要回推原圖座標。

## 面積

OpenAI 負責：

- 判斷空間
- 回傳 Polygon
- 嘗試辨識一條圖面尺寸

Python 負責：

- 使用 Shoelace Formula 計算 Polygon 像素面積
- 依 `pixels_per_meter` 換算平方公尺

若 OpenAI 無法可靠辨識比例尺，程式不會猜測平方公尺，只顯示像素面積。
