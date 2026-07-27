# 單一可編輯平面圖版本

## 本版目標

畫面只顯示一張平面圖：

```text
同一個 Fabric Canvas
├─ 最底層：不可選取的平面圖底圖
└─ 上層：可移動、拉伸、改色及刪除的 AI 多邊形
```

不再顯示：

- 額外空白底圖檢查
- PIL 座標診斷圖
- 另一張沒有底圖的 Canvas

## 底圖處理

底圖不再使用 `st_canvas(background_image=...)`，而是轉成 base64 PNG，
作為 Fabric Image 物件放進 `initial_drawing.objects` 的第一層。

底圖設定：

```text
selectable = false
evented = false
```

因此底圖與框線會在同一個畫布顯示，但底圖不能被移動或刪除。

## AI 辨識

改成兩階段：

1. 完整建築圖辨識房間名稱及大致 bbox。
2. 每個房間局部裁切，再由 GPT 精修內牆多邊形。

這比直接要求 GPT 在整張圖上一次輸出全部 polygon 更穩定。

## 更新檔案

```text
app.py
openai_room_detector.py
geometry_utils.py
requirements.txt
README.md
```
