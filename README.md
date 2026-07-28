# v1.3：多步驟 AI 空間辨識

這版保留原有 Streamlit、裁切、固定畫布及座標回推架構，只修改「房間辨識 API 呼叫方式」。

## 新流程

```text
PDF
→ 300 DPI
→ AI 找建築主體
→ 裁切與放大
→ AI 第 1 次：盤點有哪些完整室內空間，不輸出座標
→ AI 第 2~N 次：每個空間分別取得 bbox
→ AI 最後 1 次：審查候選框，刪除家具、設備、樓梯及錯誤框
→ bbox 轉 Polygon
→ 回推原圖座標
→ floorplan_editor
```

## 替換方式

只需用本版的：

- `room_bbox_detector.py`

覆蓋現有同名檔案。

`app_bbox_example.py` 與 `floorplan_editor.py` 可先保持不變。

## API 呼叫次數

假設盤點出 6 個空間，第二階段約會使用：

- 1 次：空間盤點
- 6 次：逐空間框選
- 1 次：結果審查

合計約 8 次，再加上前面的建築主體裁切辨識 1 次。

## 除錯資料

`room_detection` 現在包含：

```python
{
    "inventory": [...],
    "candidates_before_review": [...],
    "rooms": [...]
}
```

可分別檢查：

- AI 認為圖面有哪些空間
- 每個空間初次取得的 bbox
- Reviewer 最後保留的框
```
