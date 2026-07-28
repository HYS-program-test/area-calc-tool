# 平面圖 AI 視覺修正迴圈 v3 fixed

這版只修正比例與座標，不增加上下左右裁切控制。

## 關鍵修正

- `app.py` 呼叫：
  `run_visual_review_loop(original_image=original, ...)`
- `visual_review_loop.py` 函式定義也使用：
  `def run_visual_review_loop(original_image, ...)`

因此不會再發生：

```text
TypeError: run_visual_review_loop() got an unexpected keyword argument 'original_image'
```

## 座標方式

- AI：0～1000 標準化座標
- Python：轉成目前建築主體圖片的實際像素
- 不另外裁切
- 不新增上下左右界控制

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```
