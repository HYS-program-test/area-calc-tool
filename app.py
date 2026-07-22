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
st.caption("上傳平面圖 → 矩形拖曳 或 多邊形點角點 → 封閉空間 → 計算實際面積")

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
RENDER_DPI = 144          # PDF 轉圖片時的渲染解析度（fitz Matrix(2,2) 基準 72dpi）
MAX_CANVAS_WIDTH = 1000   # 顯示圖片最大寬度
PING_PER_M2 = 3.3058      # 1 坪 = 3.3058 m²
DEFAULT_COLOR = "#FF6347"

# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "last_file_key": None,
        "current_points": [],       # 多邊形模式：目前正在點選、尚未封閉的角點
        "finished_shapes": [],      # 已封閉的空間清單，每筆 {"points":[(x,y),...], "color":(b,g,r)}
        "last_click_xy": None,
        "last_drag_xy": None,
        "final_results": None,
        "final_overlay": None,
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
    st.session_state["last_drag_xy"] = None
    st.session_state["final_results"] = None
    st.session_state["final_overlay"] = None
    st.session_state["claude_review"] = None

def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

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

def draw_working_image(base_arr: np.ndarray, draw_mode: str, current_color_bgr) -> np.ndarray:
    """把已封閉的空間 + 正在畫的當前空間，疊到底圖上，讓使用者看到目前的進度"""
    arr = base_arr.copy()

    for i, shape in enumerate(st.session_state["finished_shapes"]):
        color = shape["color"]
        pts_np = np.array(shape["points"], dtype=np.int32)
        cv2.polylines(arr, [pts_np], True, color, 3)
        cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))
        cv2.circle(arr, (cx, cy), 14, color, -1)
        cv2.putText(arr, str(i + 1), (cx - 7, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if draw_mode == "多邊形":
        cur = st.session_state["current_points"]
        if cur:
            for p in cur:
                cv2.circle(arr, (int(p[0]), int(p[1])), 5, current_color_bgr, -1)
            if len(cur) > 1:
                pts_np = np.array(cur, dtype=np.int32)
                cv2.polylines(arr, [pts_np], False, current_color_bgr, 2)

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
    disp_img, display_scale = resize_display_cached(img, file_key, MAX_CANVAS_WIDTH)
    disp_arr_base = np.array(disp_img)  # 底圖的 numpy 陣列，快取起來重複使用

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
                              help="矩形：直接拖曳一個對角到另一個對角。多邊形：依序點擊每個角點，適合 L 型、斜牆等不規則空間。")
    with tool_col2:
        shape_color_hex = st.color_picker("邊框顏色", DEFAULT_COLOR)
    with tool_col3:
        if st.button("↩️ 復原上一點", use_container_width=True,
                      disabled=(draw_mode != "多邊形") or len(st.session_state["current_points"]) == 0):
            st.session_state["current_points"].pop()
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
        st.caption("在圖上直接拖曳一個角到對角，放開滑鼠就會自動記錄成一筆矩形。")

    st.caption(f"已封閉空間：{len(st.session_state['finished_shapes'])} 個")

    # ── 顯示圖片並擷取座標（矩形用拖曳、多邊形用點擊）──────────────────────
    working_arr = draw_working_image(disp_arr_base, draw_mode, current_color_bgr)

    if draw_mode == "矩形":
        result = streamlit_image_coordinates(
            working_arr, key=f"clicker_rect_{file_key}",
            click_and_drag=True, image_format="JPEG",
        )
        if result is not None and "x1" in result:
            drag_sig = (result["x1"], result["y1"], result["x2"], result["y2"])
            if drag_sig != st.session_state["last_drag_xy"]:
                st.session_state["last_drag_xy"] = drag_sig
                x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    st.session_state["finished_shapes"].append({
                        "points": rect_pts,
                        "color": current_color_bgr,
                    })
                    st.rerun()
    else:
        click = streamlit_image_coordinates(
            working_arr, key=f"clicker_poly_{file_key}",
            click_and_drag=False, image_format="JPEG",
        )
        if click is not None and "x" in click:
            xy = (click["x"], click["y"])
            if xy != st.session_state["last_click_xy"]:
                st.session_state["last_click_xy"] = xy
                st.session_state["current_points"].append(xy)
                st.rerun()

    # ── 已封閉空間清單（可個別刪除、改顏色）──────────────────────────
    if st.session_state["finished_shapes"]:
        st.markdown("**已封閉的空間：**")
        for i, shape in enumerate(st.session_state["finished_shapes"]):
            area_m2 = polygon_area_px2(shape["points"]) * m2_per_px2_display
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

    # ── 計算面積（正式輸出結果圖）──────────────────────────
    if st.button("📐 產出面積標示結果", type="primary", use_container_width=True,
                  disabled=len(st.session_state["finished_shapes"]) == 0):
        results = []
        overlay = disp_arr_base.copy()
        for i, shape in enumerate(st.session_state["finished_shapes"]):
            poly = shape["points"]
            color = shape["color"]
            area_m2 = polygon_area_px2(poly) * m2_per_px2_display
            results.append({"id": i + 1, "area_m2": area_m2, "points": poly})

            pts_np = np.array(poly, dtype=np.int32)
            cv2.polylines(overlay, [pts_np], True, color, 3)
            cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))
            label = f"#{i+1} {area_m2:.1f}m2"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(overlay, (cx - tw//2 - 4, cy - th - 6), (cx + tw//2 + 4, cy + 4), (255, 255, 255), -1)
            cv2.putText(overlay, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 0, 0), 2)

        st.session_state["final_results"] = results
        st.session_state["final_overlay"] = overlay

    # ── 顯示結果 ──────────────────────────
    if st.session_state["final_results"]:
        st.divider()
        st.subheader("計算結果")

        results = st.session_state["final_results"]
        total_m2 = sum(r["area_m2"] for r in results)

        for r in results:
            st.write(f"空間 #{r['id']}：**{r['area_m2']:.2f} m²**（約 {r['area_m2']/PING_PER_M2:.2f} 坪）")

        st.markdown(f"### 總計：{total_m2:.2f} m²（約 {total_m2/PING_PER_M2:.2f} 坪）")

        st.image(st.session_state["final_overlay"], caption="面積標示結果", use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(st.session_state["final_overlay"]).save(buf, format="PNG")
        st.download_button(
            "⬇ 下載標示圖（供報告使用）",
            data=buf.getvalue(),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_面積標示圖.png",
            mime="image/png",
            use_container_width=True,
        )

        st.divider()
        st.markdown("**🤖 Claude 輔助核對**：對照原圖，幫忙標註每個框對應的空間、指出可疑或漏框的地方（僅供參考，不影響上面已算出的面積數字）")
        if st.button("🤖 請 Claude 協助核對", use_container_width=True):
            with st.spinner("Claude 正在對照圖面檢查中…"):
                review_text = ask_claude_review(st.session_state["final_overlay"], results)
            st.session_state["claude_review"] = review_text

        if st.session_state.get("claude_review"):
            st.info(st.session_state["claude_review"])
else:
    st.info("請先上傳一份平面圖（PDF 或圖片）開始。")
