# PDF + 圖片版 OpenAI 房間 Polygon 驗證

## 支援輸入

- PDF
- PNG
- JPG
- JPEG

上傳 PDF 後，系統會：

1. 使用 PyMuPDF 讀取 PDF。
2. 以 PDF 向量物件偵測建築本體。
3. 自動裁切建築區域。
4. 將裁切結果渲染成高解析度圖片。
5. 再送入 OpenAI 取得房間 Polygon。

## Streamlit Secrets

```toml
OPENAI_API_KEY = "你的 API Key"
```

## 啟動

```bash
pip install -r requirements.txt
streamlit run app.py
```
