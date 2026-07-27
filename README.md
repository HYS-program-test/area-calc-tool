# 顯示與座標基礎修正版

本版先處理兩個基礎問題，不改動 AI 辨識策略：

1. Canvas 底圖顯示。
2. AI 多邊形進入 Fabric.js 後的座標偏移。

## 主要修改

### geometry_utils.py

- AI 多邊形由 Fabric Path 改成 Fabric Polygon。
- Polygon 使用中心點作為 `left/top/pathOffset`。
- 完整處理移動、縮放與旋轉後的座標還原。
- 舊版 Path 加入 `pathOffset` 扣除，保留相容性。

### app.py

- Canvas 背景固定使用已載入的 RGB PNG。
- 顯示原始底圖檢查。
- 增加「座標驗證預覽」，直接用 PIL 畫框，不經 Canvas。
- Canvas key 納入底圖尺寸，避免 rerun 沿用舊背景。
- 保留含底圖 PDF 匯出。

### requirements.txt

鎖定：

```text
streamlit==1.41.1
streamlit-drawable-canvas-fix==0.9.8
```

避免新版 Streamlit 內部圖片網址機制與 Canvas 元件不相容。

## 測試判讀

### 座標驗證預覽正確、Canvas 錯誤

表示 AI 座標正確，仍是 Canvas/Fabric 顯示層問題。

### 座標驗證預覽也錯誤

表示 AI 回傳的房間位置或多邊形本身錯誤，下一階段才修改 AI 流程。

### 底圖檢查有圖、Canvas 無圖

表示問題集中在 drawable canvas 套件，而不是 PDF 轉圖或裁切。
