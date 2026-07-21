import streamlit as st
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF
from streamlit_drawable_canvas import st_canvas
import re
import io
import base64
import anthropic

st.set_page_config(page_title="平面圖面積計算工具", page_icon="📐", layout="wide")

st.title("📐 平面圖面積計算工具")
st.caption("上傳平面圖 → 系統自動框出候選空間（草稿）→ 手動調整邊界 → 計算實際面積")

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
RENDER_DPI = 144          # PDF 轉圖片時的渲染解析度（fitz Matrix(2,2) 基準 72dpi）
MAX_CANVAS_WIDTH = 1000   # 畫布顯示最大寬度（效能考量，太大畫布會很卡）
DEFAULT_MIN_AREA_M2 = 5.0 # 自動偵測的最小面積門檻，太小的雜訊區塊會被濾掉
PING_PER_M2 = 3.3058      # 1 坪 = 3.3058 m²

# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "auto_polygons": None,
        "last_file_key": None,
        "final_results": None,
        "final_overlay": None,
        "claude_review": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────
# PDF / 圖片 → 可顯示圖片，並嘗試自動偵測比例尺
# ─────────────────────────────────────────────
def load_pdf(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    # 嘗試從文字內容找比例尺標示，例如「1:100」「S=1/100」
    text = page.get_text()
    auto_scale = None
    for pattern in [r'1\s*[:：]\s*(\d+)', r'1\s*/\s*(\d+)']:
        m = re.search(pattern, text)
        if m:
            candidate = int(m.group(1))
            if 10 <= candidate <= 2000:  # 合理的比例尺範圍，避免抓到不相關的數字
                auto_scale = candidate
                break

    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img, auto_scale

def polygon_to_fabric_obj(pts, stroke="#FF6347"):
    """把 [(x,y),...] 座標轉成 streamlit-drawable-canvas 看得懂的多邊形物件格式"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    left, top = min(xs), min(ys)
    obj_points = [{"x": p[0] - left, "y": p[1] - top} for p in pts]
    return {
        "type": "polygon",
        "left": left, "top": top,
        "points": obj_points,
        "fill": "rgba(255,99,71,0.25)",
        "stroke": stroke,
        "strokeWidth": 3,
        "selectable": True,
    }

def detect_candidate_polygons(disp_img: Image.Image, m2_per_px2_display: float, min_area_m2: float):
    """在縮放後的顯示圖上跑連通元件分析，抓出候選空間的多邊形頂點（自動偵測草稿）"""
    gray = cv2.cvtColor(np.array(disp_img), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ink_d = cv2.dilate(ink, np.ones((2, 2), np.uint8), iterations=1)
    bg = cv2.bitwise_not(ink_d)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)

    h_disp, w_disp = gray.shape
    polygons = []
    for idx in range(1, num_labels):
        a = stats[idx, cv2.CC_STAT_AREA]
        area_m2 = a * m2_per_px2_display
        x, y, ww, hh = (stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP],
                         stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT])
        if area_m2 < min_area_m2:
            continue
        if ww > w_disp * 0.9 and hh > h_disp * 0.9:
            continue  # 幾乎整頁大小，通常是背景縫隙而非真正房間
        mask = (labels == idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        pts = approx.reshape(-1, 2).tolist()
        if len(pts) < 3:
            continue
        polygons.append(pts)
    return polygons

def polygon_area_px2(abs_pts):
    """Shoelace 公式計算多邊形面積（像素平方）"""
    n = len(abs_pts)
    area = 0.0
    for j in range(n):
        x1, y1 = abs_pts[j]
        x2, y2 = abs_pts[(j + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

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
    prompt = f"""這是一張建築平面圖，上面已經用彩色編號框（{id_list}）標出自動偵測到的候選空間邊界。

請你對照原圖，逐一檢查：
1. 每個編號框，依圖上的文字標示或空間配置，判斷它最可能是什麼空間（例如：房間、走道、樓梯、車道、機房等）；如果無法判斷，寫「無法判斷」
2. 如果某個編號框的形狀、範圍看起來不像一個真正獨立的空間（例如貫穿多個區域、範圍異常），請標註「⚠️ 疑似非真實空間」
3. 圖面上有沒有明顯的獨立空間「完全沒被框到」？簡短描述位置（例如：左上角、靠近樓梯旁）

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

    if is_pdf:
        img, auto_scale = load_pdf(uploaded.read())
    else:
        img = Image.open(uploaded).convert("RGB")
        auto_scale = None

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

    # ── 縮放圖片以適合畫布顯示 ──────────────────────────
    display_scale = min(1.0, MAX_CANVAS_WIDTH / img.width)
    disp_img = img.resize((int(img.width * display_scale), int(img.height * display_scale)))

    # ── 換算係數：顯示像素 → 實際公尺 ──────────────────────────
    m_per_px_at_render = (2.54 / RENDER_DPI / 100) * scale_ratio if is_pdf else None
    if m_per_px_at_render:
        m_per_px_display = m_per_px_at_render / display_scale
        m2_per_px2_display = m_per_px_display ** 2
    else:
        # 圖片檔：退而求其次，假設整張圖寬度對應使用者輸入的比例尺概念下的 96 DPI 概估
        # （準確度較低，建議之後由使用者用「已知長度校正」取代）
        m_per_px_display = (2.54 / 96 / 100) * scale_ratio / display_scale
        m2_per_px2_display = m_per_px_display ** 2

    # ── 自動偵測候選邊界（只在換新檔案時重新跑一次）──────────────────────
    if st.session_state["last_file_key"] != file_key:
        st.session_state["auto_polygons"] = detect_candidate_polygons(
            disp_img, m2_per_px2_display, DEFAULT_MIN_AREA_M2
        )
        st.session_state["last_file_key"] = file_key
        st.session_state["final_results"] = None
        st.session_state["final_overlay"] = None
        st.session_state["claude_review"] = None

    show_auto = st.checkbox("顯示自動偵測的候選邊界（草稿，可再手動調整／刪除）", value=True)
    min_area_filter = st.slider("自動偵測最小面積門檻（m²）", 1.0, 30.0, DEFAULT_MIN_AREA_M2, 0.5)

    if min_area_filter != DEFAULT_MIN_AREA_M2:
        filtered_polys = detect_candidate_polygons(disp_img, m2_per_px2_display, min_area_filter)
    else:
        filtered_polys = st.session_state["auto_polygons"]

    initial_objs = [polygon_to_fabric_obj(p) for p in filtered_polys] if show_auto else []
    initial_drawing = {"version": "4.4.0", "objects": initial_objs}

    st.markdown(
        "**在下方圖面上調整邊界：**「多邊形」模式可以逐點點出新的空間（連續點擊描邊、"
        "雙擊或點回起點結束）；「選取／調整」模式可以點選既有的框，拖曳邊界，或按 Delete 鍵刪除。"
    )
    mode_choice = st.radio("畫布模式", ["🖊️ 多邊形（新增）", "✋ 選取／調整（移動、刪除）"], horizontal=True)
    drawing_mode = "polygon" if mode_choice.startswith("🖊️") else "transform"

    canvas_result = st_canvas(
        fill_color="rgba(255,99,71,0.25)",
        stroke_width=3,
        stroke_color="#FF6347",
        background_image=disp_img,
        update_streamlit=True,
        height=disp_img.height,
        width=disp_img.width,
        drawing_mode=drawing_mode,
        initial_drawing=initial_drawing,
        key=f"canvas_{file_key}",
    )

    # ── 計算面積 ──────────────────────────
    if st.button("📐 計算面積", type="primary", use_container_width=True):
        objs = []
        if canvas_result.json_data is not None:
            objs = canvas_result.json_data.get("objects", [])

        results = []
        overlay = np.array(disp_img).copy()
        for i, obj in enumerate(objs):
            if obj.get("type") != "polygon":
                continue
            left, top = obj.get("left", 0), obj.get("top", 0)
            scale_x, scale_y = obj.get("scaleX", 1), obj.get("scaleY", 1)
            pts = obj.get("points", [])
            abs_pts = [(left + p["x"] * scale_x, top + p["y"] * scale_y) for p in pts]
            if len(abs_pts) < 3:
                continue

            area_px2 = polygon_area_px2(abs_pts)
            area_m2 = area_px2 * m2_per_px2_display
            results.append({"id": i + 1, "area_m2": area_m2, "points": abs_pts})

            pts_np = np.array(abs_pts, dtype=np.int32)
            cv2.polylines(overlay, [pts_np], True, (255, 99, 71), 3)
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
