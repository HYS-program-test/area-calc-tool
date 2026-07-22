import streamlit as st
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF
from streamlit_image_coordinates import streamlit_image_coordinates
import re
import io
import base64
import anthropic

st.set_page_config(page_title="平面圖面積計算工具", page_icon="📐", layout="wide")

st.title("📐 平面圖面積計算工具")
st.caption("上傳平面圖 → 矩形點兩角 或 多邊形點角點 → 即時算出面積")

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
RENDER_DPI = 144          # PDF 轉圖片時的渲染解析度（fitz Matrix(2,2) 基準 72dpi）
MAX_CANVAS_WIDTH = 1000   # 顯示圖片最大寬度
PING_PER_M2 = 3.3058      # 1 坪 = 3.3058 m²
DEFAULT_COLOR = "#FF6347"
CROP_PADDING = 25         # 自動裁切建築物範圍時，四周多留的邊界（原圖像素）

# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "last_file_key": None,
        "current_points": [],       # 目前正在點選、尚未封閉的角點（矩形模式最多2個，多邊形不限）
        "finished_shapes": [],      # 已封閉的空間清單，每筆 {"points":[(x,y),...], "color":(b,g,r)}
        "last_click_xy": None,
        "claude_review": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def reset_drawing_state():
    st.session_state["current_points"] = []
    st.session_state["finished_shapes"] = []
    st.session_state["last_click_xy"] = None
    st.session_state["claude_review"] = None

def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

# ─────────────────────────────────────────────
# PDF / 圖片 → 可顯示圖片，並嘗試自動偵測比例尺
# 用 st.cache_data 快取:同一個檔案不會因為每次點擊都重新解析 PDF，
# 這是原本操作「每點一下就卡頓」的主要原因。
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pdf_cached(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    text = page.get_text()
    auto_scale = None
    for pattern in [r'1\s*[:：]\s*(\d+)', r'1\s*/\s*(\d+)']:
        m = re.search(pattern, text)
        if m:
            candidate = int(m.group(1))
            if 10 <= candidate <= 2000:
                auto_scale = candidate
                break

    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img, auto_scale

@st.cache_data(show_spinner=False)
def load_image_cached(file_bytes: bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

@st.cache_data(show_spinner=False)
def crop_to_content_cached(_img: Image.Image, file_key: str):
    """自動裁切掉圖面四周的大片空白（例如標題欄、圖框邊界），
    讓建築物本體置中、盡量填滿畫面，方便框選操作。"""
    gray = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return _img, (0, 0)
    x0, x1 = max(0, xs.min() - CROP_PADDING), min(_img.width, xs.max() + CROP_PADDING)
    y0, y1 = max(0, ys.min() - CROP_PADDING), min(_img.height, ys.max() + CROP_PADDING)
    cropped = _img.crop((x0, y0, x1, y1))
    return cropped, (x0, y0)

@st.cache_data(show_spinner=False)
def resize_display_cached(_img: Image.Image, file_key: str, max_width: int):
    """快取縮圖結果，同一份檔案不用每次點擊都重新resize。"""
    scale = min(1.0, max_width / _img.width)
    resized = _img.resize((int(_img.width * scale), int(_img.height * scale)))
    return resized, scale

def polygon_area_px2(pts):
    """Shoelace 公式計算多邊形面積（像素平方），支援任意不規則多邊形，矩形也適用"""
    n = len(pts)
    area = 0.0
    for j in range(n):
        x1, y1 = pts[j]
        x2, y2 = pts[(j + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def draw_all(base_arr: np.ndarray, draw_mode: str, current_color_bgr, m2_per_px2: float) -> np.ndarray:
    """畫出目前所有狀態：已封閉空間(含面積標示) + 正在畫的當前點，單一張圖同時做為
    互動用的畫布跟最終結果圖，不會有「兩張長得一樣的圖」重複出現的狀況。"""
    arr = base_arr.copy()

    for i, shape in enumerate(st.session_state["finished_shapes"]):
        color = shape["color"]
        pts_np = np.array(shape["points"], dtype=np.int32)
        cv2.polylines(arr, [pts_np], True, color, 3)
        cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))

        area_m2 = polygon_area_px2(shape["points"]) * m2_per_px2
        label = f"#{i+1} {area_m2:.1f}m2"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(arr, (cx - tw//2 - 5, cy - th - 6), (cx + tw//2 + 5, cy + 6), (255, 255, 255), -1)
        cv2.rectangle(arr, (cx - tw//2 - 5, cy - th - 6), (cx + tw//2 + 5, cy + 6), color, 2)
        cv2.putText(arr, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 正在畫的角點（不管矩形或多邊形，第一下點擊就要立刻看到標記）
    cur = st.session_state["current_points"]
    if cur:
        for p in cur:
            cv2.circle(arr, (int(p[0]), int(p[1])), 6, current_color_bgr, -1)
            cv2.circle(arr, (int(p[0]), int(p[1])), 6, (255, 255, 255), 2)
        if draw_mode == "多邊形" and len(cur) > 1:
            pts_np = np.array(cur, dtype=np.int32)
            cv2.polylines(arr, [pts_np], False, current_color_bgr, 2)
        elif draw_mode == "矩形" and len(cur) == 1:
            # 只點了第一角時，先畫出十字參考線，讓使用者知道目前點到哪
            x, y = int(cur[0][0]), int(cur[0][1])
            cv2.line(arr, (x, 0), (x, arr.shape[0]), current_color_bgr, 1)
            cv2.line(arr, (0, y), (arr.shape[1], y), current_color_bgr, 1)

    return arr

def ask_claude_review(overlay_img: np.ndarray, results: list) -> str:
    """把畫好編號框的圖交給 Claude 視覺辨識，請它幫忙標註每個框對應的空間、
    並指出看起來可疑（不像真實房間）或明顯漏框的地方。
    這是「語意判斷」輔助，不是面積計算本身——面積數字仍以程式的幾何運算為準。"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ 尚未設定 ANTHROPIC_API_KEY（請至 Streamlit Cloud → Settings → Secrets 加入）"

    buf = io.BytesIO()
    Image.fromarray(overlay_img).save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()

    id_list = "、".join(f"#{r['id']}" for r in results)
    prompt = f"""這是一張建築平面圖，上面已經用彩色編號框（{id_list}）標出使用者手動框選的空間邊界。

請你對照原圖，逐一檢查：
1. 每個編號框，依圖上的文字標示或空間配置，判斷它最可能是什麼空間（例如：房間、走道、樓梯、車道、機房等）；如果無法判斷，寫「無法判斷」
2. 如果某個編號框的形狀、範圍看起來不像一個真正獨立的空間，請標註「⚠️ 疑似有誤」
3. 圖面上有沒有明顯的獨立空間「完全沒被框到」？簡短描述位置

請用條列方式回答，每個編號一行，最後補一段「遺漏空間」的說明。"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_img}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ 呼叫 Claude 發生錯誤：{e}"

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
uploaded = st.file_uploader("上傳平面圖（PDF 或圖片）", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    file_key = f"{uploaded.name}_{uploaded.size}"
    is_pdf = uploaded.name.lower().endswith(".pdf")
    file_bytes = uploaded.getvalue()  # getvalue() 不會消耗串流，可重複使用、方便快取

    if is_pdf:
        img, auto_scale = load_pdf_cached(file_bytes)
    else:
        img = load_image_cached(file_bytes)
        auto_scale = None

    if st.session_state["last_file_key"] != file_key:
        reset_drawing_state()
        st.session_state["last_file_key"] = file_key

    # ── 自動裁切掉四周空白，讓建築物置中、盡量放大 ──────────────────────
    img_cropped, _crop_offset = crop_to_content_cached(img, file_key)

    # ── 比例尺：自動偵測 + 手動覆蓋 ──────────────────────────
    col_scale1, col_scale2 = st.columns([1, 2])
    with col_scale1:
        if auto_scale:
            st.success(f"✅ 自動偵測到比例尺 1:{auto_scale}（可在右方修正）")
        elif is_pdf:
            st.warning("⚠️ 未在圖面文字中偵測到比例尺，請手動輸入")
        else:
            st.warning("⚠️ 圖片檔無法自動偵測比例尺，請手動輸入（僅支援 PDF 自動偵測）")
    with col_scale2:
        scale_ratio = st.number_input(
            "比例尺（輸入 1:N 裡的 N）", min_value=1, value=auto_scale or 100, step=10,
            help="例如圖面是 1:100，這裡就輸入 100"
        )

    if not is_pdf:
        st.info("圖片檔沒有內建的解析度資訊，面積換算的準確度會比 PDF 差，建議優先使用 PDF。")

    # ── 縮放圖片以適合顯示（快取，不會每次點擊都重新計算）──────────────────────
    disp_img, display_scale = resize_display_cached(img_cropped, file_key, MAX_CANVAS_WIDTH)
    disp_arr_base = np.array(disp_img)

    # ── 換算係數：顯示像素 → 實際公尺 ──────────────────────────
    m_per_px_at_render = (2.54 / RENDER_DPI / 100) * scale_ratio if is_pdf else None
    if m_per_px_at_render:
        m_per_px_display = m_per_px_at_render / display_scale
    else:
        m_per_px_display = (2.54 / 96 / 100) * scale_ratio / display_scale
    m2_per_px2_display = m_per_px_display ** 2

    # ── 工具列：模式切換、顏色選擇 ──────────────────────────
    tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([1.3, 1, 1, 1])
    with tool_col1:
        draw_mode = st.radio("框選模式", ["矩形", "多邊形"], horizontal=True,
                              help="矩形：點第一個角、再點對角，自動完成。多邊形：依序點擊每個角點，適合 L 型、斜牆等不規則空間。")
    with tool_col2:
        shape_color_hex = st.color_picker("邊框顏色", DEFAULT_COLOR)
    with tool_col3:
        if st.button("↩️ 復原上一點", use_container_width=True,
                      disabled=len(st.session_state["current_points"]) == 0):
            st.session_state["current_points"].pop()
            st.rerun()
    with tool_col4:
        if st.button("🗑️ 清空全部重來", use_container_width=True):
            reset_drawing_state()
            st.rerun()

    current_color_bgr = hex_to_bgr(shape_color_hex)

    if draw_mode == "多邊形":
        st.caption("在圖上依序點擊每個角點；點完按下方「✅ 封閉此空間」記錄成一筆。")
        if st.button("✅ 封閉此空間", use_container_width=True,
                      disabled=len(st.session_state["current_points"]) < 3):
            st.session_state["finished_shapes"].append({
                "points": st.session_state["current_points"],
                "color": current_color_bgr,
            })
            st.session_state["current_points"] = []
            st.rerun()
    else:
        st.caption("在圖上點第一個角（會立刻出現標記＋參考線），再點對角，就會自動完成一筆矩形。")

    st.caption(f"已封閉空間：{len(st.session_state['finished_shapes'])} 個")

    # ── 顯示圖片並擷取點擊座標（單一張圖，即時反映所有狀態）──────────────────
    working_arr = draw_all(disp_arr_base, draw_mode, current_color_bgr, m2_per_px2_display)
    click = streamlit_image_coordinates(
        working_arr, key=f"clicker_{draw_mode}_{file_key}",
        click_and_drag=False, image_format="JPEG",
    )

    if click is not None and "x" in click:
        xy = (click["x"], click["y"])
        if xy != st.session_state["last_click_xy"]:
            st.session_state["last_click_xy"] = xy
            st.session_state["current_points"].append(xy)

            if draw_mode == "矩形" and len(st.session_state["current_points"]) == 2:
                (x1, y1), (x2, y2) = st.session_state["current_points"]
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    st.session_state["finished_shapes"].append({
                        "points": rect_pts,
                        "color": current_color_bgr,
                    })
                st.session_state["current_points"] = []

            st.rerun()

    # ── 已封閉空間清單（可個別刪除）＋ 總計 ──────────────────────────
    if st.session_state["finished_shapes"]:
        st.markdown("**已封閉的空間：**")
        total_m2 = 0.0
        for i, shape in enumerate(st.session_state["finished_shapes"]):
            area_m2 = polygon_area_px2(shape["points"]) * m2_per_px2_display
            total_m2 += area_m2
            b, g, r = shape["color"]
            c1, c2, c3 = st.columns([0.6, 4.4, 1])
            with c1:
                st.markdown(
                    f"<div style='width:20px;height:20px;border-radius:4px;background:rgb({r},{g},{b});margin-top:6px'></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.write(f"#{i+1}　約 {area_m2:.2f} m²（{len(shape['points'])} 個角點）")
            with c3:
                if st.button("刪除", key=f"del_shape_{i}"):
                    st.session_state["finished_shapes"].pop(i)
                    st.rerun()

        st.markdown(f"### 總計：{total_m2:.2f} m²（約 {total_m2/PING_PER_M2:.2f} 坪）")

        # 下載用的圖與 Claude 核對用的資料，直接從目前畫面狀態產生，不用另外再顯示一次圖
        results = [
            {"id": i + 1, "area_m2": polygon_area_px2(s["points"]) * m2_per_px2_display, "points": s["points"]}
            for i, s in enumerate(st.session_state["finished_shapes"])
        ]

        dl_col, claude_col = st.columns(2)
        with dl_col:
            buf = io.BytesIO()
            Image.fromarray(working_arr).save(buf, format="PNG")
            st.download_button(
                "⬇ 下載標示圖（供報告使用）",
                data=buf.getvalue(),
                file_name=f"{uploaded.name.rsplit('.',1)[0]}_面積標示圖.png",
                mime="image/png",
                use_container_width=True,
            )
        with claude_col:
            if st.button("🤖 請 Claude 協助核對", use_container_width=True):
                with st.spinner("Claude 正在對照圖面檢查中…"):
                    review_text = ask_claude_review(working_arr, results)
                st.session_state["claude_review"] = review_text

        if st.session_state.get("claude_review"):
            st.info(st.session_state["claude_review"])
else:
    st.info("請先上傳一份平面圖（PDF 或圖片）開始。")
