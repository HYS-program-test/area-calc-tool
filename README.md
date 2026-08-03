# 空調負荷計算

請將下列檔案全部放在 GitHub 專案根目錄：

```text
app.py
equipment.py
hvac.py
schemas.py
floorplan_editor.py
openai_gateway.py
pdf_utils.py
requirements.txt
大金空調價格表_設備報價單.xlsx

frontend/
  index.html
```

程式會自動讀取固定檔名：

```text
大金空調價格表_設備報價單.xlsx
```

設備表欄位：

- A：類別
- B：類型
- C：型號
- G：冷氣能力 kW
- I：連結機型1
- J：連結機型2
- K：連結機型3

A 欄支援：

- VRV室外機
- VRV內機
- VRV室內機
