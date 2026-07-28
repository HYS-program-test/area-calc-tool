# 平面圖框選 v1.1：加入裁切與座標回推

這版不是大幅重寫，而是在上一版補回兩個必要步驟：

1. **先找建築主體並裁切**
2. **保留 Canvas 與原圖兩套座標**

## 實際流程

```text
PDF / 圖片
↓
300 DPI 轉圖
↓
OpenAI 第一次呼叫：找建築主體 bbox
↓
回推成原圖 crop_box
↓
裁切建築主體
↓
等比例放大到 1166 × 1200 固定畫布
↓
OpenAI 第二次呼叫：框主要室內空間
↓
相對 bbox 轉 Canvas Polygon
↓
Canvas Polygon 回推原圖座標
↓
送入既有 floorplan_editor
```

## 替換方式

將以下兩個檔案放入原專案：

- `room_bbox_detector.py`
- `app_bbox_example.py`

保留你目前的：

- `floorplan_editor.py`

執行：

```bash
streamlit run app_bbox_example.py
```

## 輸出座標

每一個 room 會同時包含：

```python
{
    "points": [...],           # 1166 × 1200 Canvas 座標
    "original_points": [...]   # 原始 PDF Render 圖片座標
}
```

`points` 用在頁面編輯器。

`original_points` 用在最後輸出 PDF 疊圖。

## 注意

這版使用兩次 OpenAI 影像呼叫：

- 第一次只找建築主體，用於裁切。
- 第二次只分析裁切後的建築圖，用於房間框選。

這比把整張基地圖直接縮小後一次辨識，更接近先前人工框選流程。
