# 平面圖 AI 視覺修正迴圈

本版實作：

```text
原始圖
→ AI 產生初始 Polygon
→ Python 畫出紅框與藍色角點編號
→ AI 同時查看原圖與疊圖
→ AI 僅回傳有限修正命令
→ Python 套用命令
→ 再次畫圖與審查
→ 最多重複 1～5 輪
```

## 核心差異

Reviewer 不可重新輸出整套 Polygon，只能使用：

- approve
- move_point
- add_point
- delete_point
- delete_room
- rename_room

這可降低每輪重新猜測座標造成的漂移。

## 檔案

- `app.py`
- `visual_review_loop.py`
- `requirements.txt`
- `README.md`

## Streamlit Secrets

```toml
OPENAI_API_KEY = "你的實際 API Key"
OPENAI_VISION_MODEL = "gpt-4.1"
```

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API 次數

若設定 3 輪：

- 1 次初始候選辨識
- 最多 3 次視覺審查

合計最多 4 次 API 呼叫。

## 注意

此流程忠實複製「先畫、再看疊圖、只修改錯誤點」的修正方法，
但仍需要實際圖面測試才能判斷精度。
