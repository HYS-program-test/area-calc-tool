# Area Calc Tool 修正版

本包修正：

- `run_visual_review_loop()` 改用位置參數呼叫。
- 不再傳入 `original_image=`。
- 介面只顯示建築主體。
- 不增加上下左右界控制。
- AI、格線與 Polygon 全部使用同一張建築主體圖片。
- 座標統一使用 0～1000 標準化座標。

請同時覆蓋：

- `app.py`
- `visual_review_loop.py`
- `requirements.txt`

更新後請 Reboot Streamlit App。
