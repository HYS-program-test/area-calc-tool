import streamlit as st
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF
import streamlit_drawable_canvas as _sdc_module
from streamlit_drawable_canvas import CanvasResult
import re
import io
import base64
import anthropic

st.set_page_config(page_title="平面圖面積計算工具", page_icon="📐", layout="wide")

RENDER_DPI = 144
MAX_CANVAS_WIDTH = 1100
PING_PER_M2 = 3.3058
CROP_PADDING = 25
FIXED_COLORS = ["#FF6347", "#3B82F6", "#22C55E", "#F59E0B", "#A855F7", "#06B6D4"]

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {"last_file_key": None, "canvas_json": None, "claude_review": None, "color_idx": 0}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def reset_drawing_state():
    st.session_state["canvas_json"] = None
    st.session_state["claude_review"] = None

# ─────────────────────────────────────────────
# 安全版 st_canvas：自己組 base64 網址傳背景圖，繞過套件內部
# image_to_url() 在部分情況下失效、導致背景圖顯示不出來的問題
# （這是先前實測過確實有效的修正）
# ─────────────────────────────────────────────
def st_canvas_safe(fill_color, stroke_width, stroke_color, background_image,
                    height, width, drawing_mode, initial_drawing, key):
    buf = io.BytesIO()
    background_image.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    background_image_url = f"data:image/jpeg;base64,{b64}"

    initial_drawing = {"version": "4.4.0"} if initial_drawing is None else dict(initial_drawing)
    initial_drawing["background"] = ""

    try:
        component_value = _sdc_module._component_func(
            fillColor=fill_color, strokeWidth=stroke_width, strokeColor=stroke_color,
            backgroundColor="", backgroundImageURL=background_image_url,
            realtimeUpdateStreamlit=True,
            canvasHeight=height, canvasWidth=width,
            drawingMode=drawing_mode, initialDrawing=initial_drawing,
            displayToolbar=True, displayRadius=3, key=key, default=None,
        )
    except Exception as e:
        st.error(f"畫布元件載入失敗：{e}")
        return CanvasResult()

    if component_value is None:
        return CanvasResult()
    return CanvasResult(
        np.asarray(_sdc_module._data_url_to_image(component_value["data"])),
        component_value["raw"],
    )

# ─────────────────────────────────────────────
# 圖片載入 / 裁切 / 縮放（快取）
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
    gray = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    ink_d = cv2.dilate(ink, np.ones((5, 5), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_d, connectivity=8)
    if n <= 1:
        return _img
    best_idx, best_score = None, -1
    for idx in range(1, n):
        x, y, w, h, area = stats[idx]
        bbox_area = w * h
        if bbox_area == 0:
            continue
        score = (area / bbox_area) * area
        if score > best_score:
            best_score, best_idx = score, idx
    if best_idx is None:
        return _img
    x, y, w, h, _ = stats[best_idx]
    x0, y0 = max(0, x - CROP_PADDING), max(0, y - CROP_PADDING)
    x1, y1 = min(_img.width, x + w + CROP_PADDING), min(_img.height, y + h + CROP_PADDING)
    return _img.crop((x0, y0, x1, y1))

@st.cache_data(show_spinner=False)
def resize_display_cached(_img: Image.Image, file_key: str, max_width: int):
    scale = min(1.0, max_width / _img.width)
    resized = _img.resize((int(_img.width * scale), int(_img.height * scale)))
    return resized, scale

def polygon_area_px2(pts):
    n = len(pts)
    area = 0.0
    for j in range(n):
        x1, y1 = pts[j]
        x2, y2 = pts[(j + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def extract_shapes(json_data):
    """從畫布的 json_data 解析出每個物件的實際頂點座標（矩形／多邊形皆轉成頂點清單）"""
    if not json_data:
        return []
    shapes = []
    for obj in json_data.get("objects", []):
        color = obj.get("stroke", "#FF6347")
        left, top = obj.get("left", 0), obj.get("top", 0)
        scale_x, scale_y = obj.get("scaleX", 1), obj.get("scaleY", 1)
        angle = obj.get("angle", 0)
        if obj.get("type") == "rect":
            w, h = obj.get("width", 0) * scale_x, obj.get("height", 0) * scale_y
            pts = [(0, 0), (w, 0), (w, h), (0, h)]
        elif obj.get("type") == "polygon":
            pts = [(p["x"] * scale_x, p["y"] * scale_y) for p in obj.get("points", [])]
        else:
            continue
        # 套用旋轉角度（若有被轉過）
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        abs_pts = [(left + x*cos_t - y*sin_t, top + x*sin_t + y*cos_t) for x, y in pts]
        if len(abs_pts) >= 3:
            shapes.append({"points": abs_pts, "color": color})
    return shapes

def ask_claude_review(overlay_img: np.ndarray, results: list) -> str:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ 尚未設定 ANTHROPIC_API_KEY"
    buf = io.BytesIO()
    Image.fromarray(overlay_img).save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()
    id_list = "、".join(f"#{r['id']}" for r in results)
    prompt = f"""這是一張建築平面圖，上面已經用彩色編號框（{id_list}）標出使用者手動框選的空間邊界。

請你對照原圖，逐一檢查：
1. 每個編號框，依圖上的文字標示或空間配置，判斷它最可能是什麼空間；如果無法判斷，寫「無法判斷」
2. 如果某個編號框的形狀、範圍看起來不像一個真正獨立的空間，請標註「⚠️ 疑似有誤」
3. 圖面上有沒有明顯的獨立空間「完全沒被框到」？簡短描述位置

請用條列方式回答，每個編號一行，最後補一段「遺漏空間」的說明。"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_img}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ 呼叫 Claude 發生錯誤：{e}"

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
title_col, upload_col = st.columns([1.2, 2.5])
with title_col:
    st.markdown("##### 📐 平面圖面積計算工具")
with upload_col:
    uploaded = st.file_uploader("上傳平面圖", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded:
    file_key = f"{uploaded.name}_{uploaded.size}"
    is_pdf = uploaded.name.lower().endswith(".pdf")
    file_bytes = uploaded.getvalue()

    if is_pdf:
        img, auto_scale = load_pdf_cached(file_bytes)
    else:
        img = load_image_cached(file_bytes)
        auto_scale = None

    if st.session_state["last_file_key"] != file_key:
        reset_drawing_state()
        st.session_state["last_file_key"] = file_key

    img_cropped = crop_to_content_cached(img, file_key)
    disp_img, display_scale = resize_display_cached(img_cropped, file_key, MAX_CANVAS_WIDTH)

    t1, t2, t3, t4 = st.columns([1.3, 1.4, 1.6, 1.6])
    with t1:
        scale_ratio = st.number_input(
            f"比例尺 1:N｜{'✅自動' if auto_scale else '⚠️手動'}",
            min_value=1, value=auto_scale or 100, step=10,
        )
    with t2:
        draw_mode = st.radio("模式", ["矩形", "多邊形", "選取／調整"], horizontal=True)
    with t3:
        color_labels = ["🔴", "🔵", "🟢", "🟠", "🟣", "🔷"]
        picked = st.radio("顏色", color_labels, horizontal=True,
                           index=st.session_state["color_idx"])
        st.session_state["color_idx"] = color_labels.index(picked)
        shape_color_hex = FIXED_COLORS[st.session_state["color_idx"]]
    with t4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ 清空全部重來", use_container_width=True):
            reset_drawing_state()
            st.rerun()

    if not is_pdf:
        st.caption("⚠️ 圖片檔沒有內建解析度資訊，面積換算準確度會比 PDF 差。")

    m_per_px_at_render = (2.54 / RENDER_DPI / 100) * scale_ratio if is_pdf else None
    if m_per_px_at_render:
        m_per_px_display = m_per_px_at_render / display_scale
    else:
        m_per_px_display = (2.54 / 96 / 100) * scale_ratio / display_scale
    m2_per_px2_display = m_per_px_display ** 2

    mode_map = {"矩形": "rect", "多邊形": "polygon", "選取／調整": "transform"}

    canvas_result = st_canvas_safe(
        fill_color=shape_color_hex + "40",
        stroke_width=3,
        stroke_color=shape_color_hex,
        background_image=disp_img,
        height=disp_img.height,
        width=disp_img.width,
        drawing_mode=mode_map[draw_mode],
        initial_drawing=st.session_state["canvas_json"],
        key=f"canvas_{file_key}",
    )

    # 官方標準雙向同步：畫布每次變動，realtimeUpdateStreamlit=True 就會自動把最新
    # json_data 傳回來，這裡存進 session_state，下次重繪（例如切換模式）時當作
    # initial_drawing 餵回去，藏框不會因為切模式而消失
    if canvas_result.json_data is not None:
        st.session_state["canvas_json"] = canvas_result.json_data

    shapes = extract_shapes(st.session_state["canvas_json"])

    # ── 結果 ──────────────────────────
    if shapes:
        total_m2 = sum(polygon_area_px2(s["points"]) * m2_per_px2_display for s in shapes)
        res_cols = st.columns(min(len(shapes), 8) + 1)
        for i, shape in enumerate(shapes[:8]):
            area_m2 = polygon_area_px2(shape["points"]) * m2_per_px2_display
            color = shape["color"]
            with res_cols[i]:
                st.markdown(
                    f"<div style='text-align:center;padding:4px;border-radius:6px;background:{color}22;border:1px solid {color}'>"
                    f"<b style='color:{color}'>#{i+1}</b><br>{area_m2:.2f} m²</div>",
                    unsafe_allow_html=True,
                )
        with res_cols[-1]:
            st.markdown(
                f"<div style='text-align:center;padding:4px;border-radius:6px;background:#f0f4ff;border:1px solid #1a3f6f'>"
                f"<b>總計</b><br>{total_m2:.2f} m²</div>", unsafe_allow_html=True,
            )
        st.caption(f"約 {total_m2/PING_PER_M2:.2f} 坪")

        with st.expander("🤖 Claude 輔助核對（對照原圖標註每個框對應的空間，僅供參考）"):
            results = [
                {"id": i + 1, "area_m2": polygon_area_px2(s["points"]) * m2_per_px2_display, "points": s["points"]}
                for i, s in enumerate(shapes)
            ]
            if st.button("請 Claude 協助核對"):
                overlay = np.array(disp_img).copy()
                for i, shape in enumerate(shapes):
                    pts_np = np.array(shape["points"], dtype=np.int32)
                    hexc = shape["color"].lstrip("#")
                    r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
                    cv2.polylines(overlay, [pts_np], True, (r, g, b), 3)
                    cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))
                    cv2.putText(overlay, f"#{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                with st.spinner("Claude 正在對照圖面檢查中…"):
                    review_text = ask_claude_review(overlay, results)
                st.session_state["claude_review"] = review_text
            if st.session_state.get("claude_review"):
                st.info(st.session_state["claude_review"])
    else:
        st.caption("尚未框選任何空間，請在上方圖面開始框選。")
else:
    st.info("請先上傳一份平面圖（PDF 或圖片）開始。")
