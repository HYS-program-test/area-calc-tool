# OpenAI 房間 Polygon 真實驗證

在 Streamlit Cloud 的 App settings → Secrets 加入：

```toml
OPENAI_API_KEY = "你的 API Key"
```

不要把 API Key 寫入 GitHub。

執行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

上傳 `2F_building_crop.png`，按下「呼叫 OpenAI 並框選房間」。

這個測試驗證：
- OpenAI 是否能辨識房間
- 是否能回傳 Polygon
- Polygon 疊回圖片的位置是否合理

像素面積尚未換算平方公尺。
