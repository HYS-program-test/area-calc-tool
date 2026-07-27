# 單一畫布 v4：WebSocket 與框選修正

## WebSocket 修正

上一版將整張底圖轉成 base64，並存入 `st.session_state.drawing`。
每次拖曳框線時，Streamlit 都可能透過 WebSocket 重送大型 JSON，
造成：

```text
Cached ForwardMsg MISS
框線閃爍
Connection error
```

本版改為：

```text
background_image = 壓縮 JPEG PIL 圖片
session_state = 只保存小型 polygon JSON
```

底圖不再進入 session_state，也不再反覆回傳 base64。

## 框選修正

GPT 不再直接輸出 polygon，只輸出：

```text
空間名稱
空間類型
大致 bbox
```

再由 OpenCV 以 bbox 中心作 seed，在局部範圍尋找封閉空白區域並轉成 polygon。

若局部分割失敗，使用 GPT bbox 作可編輯矩形，而不是接受 GPT 產生的巨大錯誤 polygon。

## 版本

```text
streamlit==1.49.1
streamlit-drawable-canvas-fix==0.9.8
```
