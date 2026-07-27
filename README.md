# AI 平面圖空間辨識與空調設備選型

本版已將主要流程從 OpenCV 規則式封閉空間偵測，改為：

```text
平面圖
→ OpenAI Vision 直接辨識房間
→ 回傳正規化多邊形
→ 轉成 Streamlit 可編輯框線
→ 人工確認
→ 比例尺校正
→ 面積與空調負荷
```

## 檔案

```text
app.py
openai_room_detector.py
openai_reviewer.py
floorplan_detector.py
geometry_utils.py
requirements.txt
```

`floorplan_detector.py` 仍保留為備援辨識與自動裁切用途。

## Streamlit Secrets

```toml
OPENAI_API_KEY = "sk-proj-..."

EQUIPMENT_SHEET_ID = "Google Sheet ID"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----
"""
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 使用注意

- AI 直接回傳的是候選多邊形，不是 CAD 測量成果。
- 必須在畫布確認內牆邊界，再做比例尺校正。
- 同一張圖重跑 AI，框線可能略有差異。
- 完整施工圖包含家具、尺寸線、門弧時，AI 通常比單純的形態學封閉空間更能理解語意，但座標仍可能有偏移。
