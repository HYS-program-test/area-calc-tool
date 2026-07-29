# 空調負荷計算整合版

功能：
- PDF / 圖片上傳
- PDF 自動裁切建築本體
- OpenAI 自動框選房間
- 預設 60% 圖面縮放
- 框框移動、拉伸、刪除、改色、新增
- 空調負荷表格即時更新
- CSV 匯出

Streamlit Cloud Secrets：

```toml
OPENAI_API_KEY = "你的 API Key"
```

啟動：

```bash
pip install -r requirements.txt
streamlit run app.py
```
