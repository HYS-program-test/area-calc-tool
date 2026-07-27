# 平面圖空調設備選型

檔案：`app.py`、`floorplan_detector.py`、`geometry_utils.py`、`openai_reviewer.py`、`requirements.txt`。

執行：
```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Secrets：
```toml
OPENAI_API_KEY = "sk-..."
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

OpenCV 負責候選空間幾何辨識；OpenAI 只負責語意複核；畫布提供拖曳、拉伸、刪除、重畫與改色。
