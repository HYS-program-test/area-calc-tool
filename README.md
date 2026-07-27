OpenCV 粗定位＋GPT 精辨識版

新版流程

PDF／圖片
→ OpenCV 找出建築主體
→ 將建築主體切成 1～9 個核心責任區
→ 每個核心區加上重疊上下文
→ GPT 分別辨識局部房間
→ 局部座標映射回完整圖面
→ 排除低信心、過小與重複候選框
→ Streamlit 人工修正
→ 比例尺校正
→ 面積與空調設備選型

為什麼改成局部辨識

完整施工圖同時包含尺寸線、家具、門弧、樓梯與文字。OpenCV 不再嘗試直接把這些線條封閉成房間，只負責定位與切圖；GPT 則在較小的局部圖中分辨真正牆體與房間。

主要檔案

app.py
openai_room_detector.py
openai_reviewer.py
floorplan_detector.py
geometry_utils.py
requirements.txt

建議起始設定

OpenAI 視覺模型：gpt-4.1
最低信心分數：0.35
GPT 分析區塊數上限：6
區塊上下文重疊率：0.18

區塊數越多，局部圖越清楚，但 API 呼叫次數會增加。一般單層住宅建議先使用 4～6 個區塊。

Streamlit Secrets

OPENAI_API_KEY = "sk-proj-..."

Google Sheets 設定沿用原版。

重要限制

GPT 回傳的是候選框，不是 CAD 測量成果。

AI 的多邊形仍可能在門洞或內牆處偏移。

必須人工確認框線並使用已知尺寸線校正比例尺。

若某一區塊呼叫失敗，程式會保留其他成功區塊的結果，並在摘要中列出錯誤。
