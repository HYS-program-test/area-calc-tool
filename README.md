# Fabric.js + GPT 逐房間辨識版

## 核心改變

### 不再使用 streamlit-drawable-canvas

改用自訂 Fabric.js Streamlit Component。
拖曳、拉伸、改色、刪除與復原都在瀏覽器內執行，
只有按「套用修改」時才把框線 JSON 傳回 Python。

因此可避免：

- 每次拖曳觸發 Streamlit rerun
- 框線閃爍
- Cached ForwardMsg MISS
- 底圖反覆載入或消失

### GPT 每張圖完整判讀

```text
完整平面圖
→ GPT 辨識房間名稱與粗略 bbox
→ 每個房間局部裁切
→ GPT 再次判斷內牆 polygon
→ 幾何驗證
→ Fabric.js 人工修正
```

## 更新檔案

```text
app.py
openai_room_detector.py
floorplan_editor.py
frontend/index.html
requirements.txt
README.md
```
