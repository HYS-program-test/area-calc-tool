import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from streamlit_image_coordinates import streamlit_image_coordinates
import re
import json
import io
import base64
import anthropic

st.set_page_config(page_title="平面圖面積計算工具", page_icon="📐", layout="wide")

RENDER_DPI = 144
PING_PER_M2 = 3.3058
CROP_PADDING = 25
FIXED_COLORS = ["#FF6347", "#3B82F6", "#22C55E", "#F59E0B", "#A855F7", "#06B6D4"]
COLOR_LABELS = ["🔴", "🔵", "🟢", "🟠", "🟣", "🔷"]
LOAD_OPTIONS = list(range(400, 1300, 100))  # 400~1200，每100一個
DEVICE_CATEGORIES = ["RA", "SA", "MA", "VRV"]

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "last_file_key": None,
        "current_points": [],
        "finished_shapes": [],   # [{"points":[(x,y),...], "color":(b,g,r)}]
        "last_click_xy": None,
        "claude_review": None,
        "claude_detect_analysis": None,
        "color_idx": 0,
        "equip_table": None,
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
    st.session_state["claude_detect_analysis"] = None

def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

# ─────────────────────────────────────────────
# 設備資料表（連結 Google Sheets「Total Certificate Management」）
# 沒設定 secrets 時優雅地回傳空清單，不會讓程式壞掉，只是下拉選單先是空的
# 欄位對照：B=類型、C=室外機、D 或 AJ=室內機、Q=室內機冷房能力
# ─────────────────────────────────────────────
def _col(row, idx):
    return row[idx].strip() if len(row) > idx and row[idx] else ""

@st.cache_data(show_spinner=False, ttl=300)
def load_equipment_data():
    """回傳 (室內機清單, 室內機->資料 查找表)。查找表的 value 是
    {"類型":..., "室外機":..., "室內機冷房能力":...}，用室內機型號查其他欄位。"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        sa_info = dict(st.secrets["gcp_service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)

        sheet_id = st.secrets.get("EQUIPMENT_SHEET_ID", "1hEt4uxBABBicxIMJuR57lMiigQYF02CQHZfB-Nc6vjo")
        sh = gc.open_by_key(sheet_id)
        ws = sh.get_worksheet(0)
        values = ws.get_all_values()

        IDX_TYPE, IDX_OUTDOOR, IDX_INDOOR_D, IDX_INDOOR_AJ, IDX_CAPACITY = 1, 2, 3, 35, 16

        lookup = {}
        for row in values[2:]:
            indoor = _col(row, IDX_INDOOR_D) or _col(row, IDX_INDOOR_AJ)
            if not indoor:
                continue
            lookup[indoor] = {
                "類型": _col(row, IDX_TYPE),
                "室外機": _col(row, IDX_OUTDOOR),
                "室內機冷房能力": _col(row, IDX_CAPACITY),
            }
        return sorted(lookup.keys()), lookup
    except Exception:
        return [], {}

# ─────────────────────────────────────────────
# 圖片載入 / 裁切 / 縮放（快取，避免每次互動都重新運算造成卡頓）
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
    """自動裁切掉圖面四周空白／外框，用「墨跡密度」找出真正的建築本體
    （排除滿版圖框線這種 bbox 很大但密度很低的東西），置中放大顯示。"""
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

def draw_all(base_arr: np.ndarray, draw_mode: str, current_color_bgr) -> np.ndarray:
    arr = base_arr.copy()
    for i, shape in enumerate(st.session_state["finished_shapes"]):
        color = shape["color"]
        pts_np = np.array(shape["points"], dtype=np.int32)
        cv2.polylines(arr, [pts_np], True, color, 3)
        cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))
        label = f"#{i+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(arr, (cx - tw//2 - 5, cy - th - 6), (cx + tw//2 + 5, cy + 6), (255, 255, 255), -1)
        cv2.rectangle(arr, (cx - tw//2 - 5, cy - th - 6), (cx + tw//2 + 5, cy + 6), color, 2)
        cv2.putText(arr, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cur = st.session_state["current_points"]
    if cur:
        for p in cur:
            cv2.circle(arr, (int(p[0]), int(p[1])), 6, current_color_bgr, -1)
            cv2.circle(arr, (int(p[0]), int(p[1])), 6, (255, 255, 255), 2)
        if draw_mode == "多邊形" and len(cur) > 1:
            pts_np = np.array(cur, dtype=np.int32)
            cv2.polylines(arr, [pts_np], False, current_color_bgr, 2)
        elif draw_mode == "矩形" and len(cur) == 1:
            x, y = int(cur[0][0]), int(cur[0][1])
            cv2.line(arr, (x, 0), (x, arr.shape[0]), current_color_bgr, 1)
            cv2.line(arr, (0, y), (arr.shape[1], y), current_color_bgr, 1)
    return arr

def safe_resize_for_claude(img: Image.Image, max_edge=1568, max_tokens=1568, patch=28):
    """依官方文件的規則，預先把圖片縮到「保證不會被 Claude 內部再次縮放」的範圍內
    （長邊上限 + 視覺 token 預算兩個限制都要顧到，直向的長圖最容易只顧到長邊、忽略 token 預算）。
    這樣送出去的圖跟 Claude 實際「看到」的圖是同一張，座標才不會對不起來。"""
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    w2, h2 = w * scale, h * scale
    tokens = (w2 / patch) * (h2 / patch)
    if tokens > max_tokens:
        area_scale = (max_tokens / tokens) ** 0.5
        w2, h2 = w2 * area_scale, h2 * area_scale
    w2, h2 = max(1, int(w2)), max(1, int(h2))
    if (w2, h2) == img.size:
        return img
    return img.resize((w2, h2))

def ask_claude_detect_rooms(disp_img: Image.Image):
    """請 Claude 直接用視覺理解去判斷平面圖上每個獨立房間的邊界。
    這是「語意判斷」出的草稿，精確度不會是像素級的，仍需要人工用框選工具核對調整。

    座標策略（依 Anthropic 官方文件建議調整過）：
    - 官方文件明確建議用「絕對像素座標」而不是相對比例，並且要先把圖片縮到 Claude
      不會再內部縮放的範圍內，兩張圖（我送的、Claude 看的）才會完全一致，座標不用再換算。
    - 直向的長圖（我們的平面圖很常見）最容易觸發「視覺 token 預算」限制而被內部縮小，
      官方文件說這是座標對不齊最常見的原因，所以這裡改成自己先用 safe_resize_for_claude()
      縮到安全範圍，並在 prompt 裡明講這張圖確切的像素尺寸，再回頭把 Claude 給的絕對座標
      按比例換算回畫面顯示用的 disp_img 尺寸。"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, "⚠️ 尚未設定 ANTHROPIC_API_KEY", None

    send_img = safe_resize_for_claude(disp_img)
    sw, sh = send_img.size

    buf = io.BytesIO()
    send_img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()

    prompt = f"""這是一張建築平面圖，圖片確切尺寸是寬 {sw} 像素、高 {sh} 像素（左上角是像素座標
(0, 0)，右下角是 ({sw}, {sh})）。請你判斷圖中每一個獨立的房間／空間（忽略樓梯間、電梯核心、
純走道這類開放式流通空間，只框出有明確機能的獨立房間，例如臥室、衛浴、廚房、客廳、儲藏室等）。

請分兩步驟回答：

【第一步：文字分析】
針對每一個房間，用文字描述它的四個邊界分別是什麼（例如：「臥室A：左邊到外牆、右邊到與臥室B
之間的隔間牆、上邊到走道的牆面、下邊到與衛浴共用的牆」）。這一步是為了讓你先仔細觀察牆面位置，
再進到第二步。

【第二步：絕對像素座標】
根據第一步的分析，把每個房間邊界轉換成角點座標——**用絕對像素座標，不要用 0~1 的相對比例**，
數值範圍就是 0 到 {sw}（寬）、0 到 {sh}（高），跟這張圖本身的像素尺寸一致。矩形房間給 4 個
角點，不規則（L型等）房間可以給更多角點，依第一步描述的實際牆角轉換。

第二步的座標結果，請用下面這兩個標記包住，標記中間只能放 JSON，不要有其他文字：
===JSON_START===
[
  {{"name": "房間名稱（看得出來就填，看不出來就填空字串）", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}},
  ...
]
===JSON_END==="""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=3072,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_img}},
                {"type": "text", "text": prompt},
            ]}],
        )
        raw_text = response.content[0].text.strip()
        m = re.search(r"===JSON_START===(.*?)===JSON_END===", raw_text, flags=re.DOTALL)
        if not m:
            return None, "⚠️ Claude 回應格式不符預期（找不到 JSON 標記），請重試", raw_text
        json_text = re.sub(r"^```(json)?|```$", "", m.group(1).strip(), flags=re.MULTILINE).strip()
        rooms = json.loads(json_text)
        # 把 Claude 給的絕對像素座標（相對於 send_img 尺寸），按比例換算回 disp_img 尺寸
        scale_x, scale_y = disp_img.width / sw, disp_img.height / sh
        for room in rooms:
            room["points"] = [[px * scale_x, py * scale_y] for px, py in room.get("points", [])]
        analysis_text = raw_text.split("===JSON_START===")[0].strip()
        return rooms, None, analysis_text
    except Exception as e:
        return None, f"⚠️ 呼叫 Claude 發生錯誤：{e}", None

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
# 版面：精簡標題列
# ─────────────────────────────────────────────
title_col, upload_col = st.columns([1.2, 2.5])
with title_col:
    st.markdown("##### 📐 平面圖面積計算工具")
with upload_col:
    uploaded = st.file_uploader("上傳平面圖", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")

shapes_for_table = []

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

    size_label = st.select_slider(
        "圖面尺寸（會影響畫面顯示、框選、以及送給 Claude 辨識用的圖，三者都是同一份）",
        options=["小 (600px)", "中 (900px)", "大 (1150px)"], value="大 (1150px)",
    )
    working_width = {"小 (600px)": 600, "中 (900px)": 900, "大 (1150px)": 1150}[size_label]

    disp_img, display_scale = resize_display_cached(img_cropped, f"{file_key}_{working_width}", working_width)
    disp_arr_base = np.array(disp_img)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.caption(f"📏 目前工作圖實際像素尺寸：{disp_img.width} × {disp_img.height} px"
                   "（畫面上的框會撐滿版面顯示，所以看起來大小差不多是正常的，這裡的數字才是真的尺寸）")

        with st.expander("🖊️ 框選工具", expanded=True):
            # ── 緊湊工具列 ──────────────────────────
            t1, t2 = st.columns([1.5, 1.5])
            with t1:
                scale_ratio = st.number_input(
                    f"比例尺 1:N｜{'✅自動' if auto_scale else '⚠️手動'}",
                    min_value=1, value=auto_scale or 100, step=10,
                )
            with t2:
                picked = st.radio("顏色", COLOR_LABELS, horizontal=True, index=st.session_state["color_idx"])
                st.session_state["color_idx"] = COLOR_LABELS.index(picked)
                shape_color_hex = FIXED_COLORS[st.session_state["color_idx"]]

            t3, t4, t5, t6 = st.columns([1.6, 1, 1, 1])
            with t3:
                draw_mode = st.radio("模式", ["矩形", "多邊形"], horizontal=True,
                                      help="矩形：點第一角、再點對角自動完成。多邊形：依序點角點，點回起點附近自動封閉。")
            with t4:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("↩️ 復原", use_container_width=True,
                              disabled=len(st.session_state["current_points"]) == 0):
                    st.session_state["current_points"].pop()
                    st.rerun()
            with t5:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("🗑️ 刪末框", use_container_width=True,
                              disabled=len(st.session_state["finished_shapes"]) == 0):
                    st.session_state["finished_shapes"].pop()
                    st.rerun()
            with t6:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("🗑️ 清空", use_container_width=True):
                    reset_drawing_state()
                    st.rerun()

            if not is_pdf:
                st.caption("⚠️ 圖片檔沒有內建解析度資訊，面積換算準確度會比 PDF 差。")

            ai_detect_clicked = st.button("🤖 Claude 自動框選（草稿）", use_container_width=True,
                                           help="請 Claude 用視覺判斷直接框出房間邊界，當作草稿，仍建議人工核對調整")
            st.caption(f"目前用「{size_label}」這份圖送給 Claude 辨識，跟顯示、框選用的是同一份圖。"
                       "Claude 判斷出的邊界是語意層級的估計，不是像素級精準測量，框好後請切到「矩形／多邊形」模式手動微調。")

            if ai_detect_clicked:
                with st.spinner(f"Claude 正在判讀平面圖（{size_label}），先分析再框邊界中…"):
                    rooms, err, analysis = ask_claude_detect_rooms(disp_img)
                if err:
                    st.error(err)
                    if analysis:
                        with st.expander("查看 Claude 原始回應（除錯用）"):
                            st.text(analysis)
                elif not rooms:
                    st.warning("Claude 沒有辨識出任何房間，請改用手動框選。")
                else:
                    new_shapes = []
                    for i, room in enumerate(rooms):
                        pts = room.get("points", [])
                        if len(pts) < 3:
                            continue
                        abs_pts = [(px, py) for px, py in pts]
                        color = hex_to_bgr(FIXED_COLORS[i % len(FIXED_COLORS)])
                        new_shapes.append({"points": abs_pts, "color": color})
                    st.session_state["finished_shapes"] = new_shapes
                    st.session_state["current_points"] = []
                    if analysis:
                        st.session_state["claude_detect_analysis"] = analysis
                    st.success(f"Claude 框出了 {len(new_shapes)} 個房間草稿，請往下核對、用框選工具調整。")
                    st.rerun()

            if st.session_state.get("claude_detect_analysis"):
                with st.expander("📝 Claude 的文字分析（框選依據，可對照檢查哪裡判斷錯了）"):
                    st.text(st.session_state["claude_detect_analysis"])

            m_per_px_at_render = (2.54 / RENDER_DPI / 100) * scale_ratio if is_pdf else None
            if m_per_px_at_render:
                m_per_px_display = m_per_px_at_render / display_scale
            else:
                m_per_px_display = (2.54 / 96 / 100) * scale_ratio / display_scale
            m2_per_px2_display = m_per_px_display ** 2
            current_color_bgr = hex_to_bgr(shape_color_hex)

            # ── 圖面：主要區域，撐滿左欄可用寬度顯示 ──────────────────────
            working_arr = draw_all(disp_arr_base, draw_mode, current_color_bgr)
            click = streamlit_image_coordinates(
                working_arr, key=f"clicker_{draw_mode}_{file_key}",
                click_and_drag=False, image_format="JPEG",
                use_column_width="always",
            )

            if click is not None and "x" in click:
                disp_w = click.get("width") or disp_img.width
                disp_h = click.get("height") or disp_img.height
                scale_x = disp_img.width / disp_w if disp_w else 1
                scale_y = disp_img.height / disp_h if disp_h else 1
                real_x = click["x"] * scale_x
                real_y = click["y"] * scale_y

                xy = (real_x, real_y)
                if xy != st.session_state["last_click_xy"]:
                    st.session_state["last_click_xy"] = xy
                    st.session_state["current_points"].append(xy)

                    if draw_mode == "矩形" and len(st.session_state["current_points"]) == 2:
                        (x1, y1), (x2, y2) = st.session_state["current_points"]
                        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                            rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            st.session_state["finished_shapes"].append({"points": rect_pts, "color": current_color_bgr})
                        st.session_state["current_points"] = []
                    elif draw_mode == "多邊形" and len(st.session_state["current_points"]) > 2:
                        x0, y0 = st.session_state["current_points"][0]
                        if ((xy[0]-x0)**2 + (xy[1]-y0)**2) ** 0.5 < 14:
                            poly_pts = st.session_state["current_points"][:-1]
                            st.session_state["finished_shapes"].append({"points": poly_pts, "color": current_color_bgr})
                            st.session_state["current_points"] = []

                    st.rerun()

            if st.session_state["finished_shapes"]:
                with st.expander("🤖 Claude 輔助核對（對照原圖標註每個框對應的空間，僅供參考）"):
                    results = [
                        {"id": i + 1, "area_m2": polygon_area_px2(s["points"]) * m2_per_px2_display, "points": s["points"]}
                        for i, s in enumerate(st.session_state["finished_shapes"])
                    ]
                    if st.button("請 Claude 協助核對"):
                        with st.spinner("Claude 正在對照圖面檢查中…"):
                            review_text = ask_claude_review(working_arr, results)
                        st.session_state["claude_review"] = review_text
                    if st.session_state.get("claude_review"):
                        st.info(st.session_state["claude_review"])

    # ── 右半部：框選後的面積結果，直向清單，不管左邊工具有沒有摺疊都一直顯示 ──────────
    with right_col:
        st.markdown("##### 📊 面積結果")
        if st.session_state["finished_shapes"]:
            total_m2 = sum(polygon_area_px2(s["points"]) * m2_per_px2_display for s in st.session_state["finished_shapes"])
            for i, shape in enumerate(st.session_state["finished_shapes"]):
                area_m2 = polygon_area_px2(shape["points"]) * m2_per_px2_display
                b, g, r = shape["color"]
                shapes_for_table.append({"name": f"#{i+1}", "area": round(area_m2, 2)})
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:8px 12px;margin-bottom:6px;border-radius:6px;"
                    f"background:rgba({r},{g},{b},0.10);border-left:4px solid rgb({r},{g},{b})'>"
                    f"<b style='color:rgb({r},{g},{b})'>#{i+1}</b>"
                    f"<span>{area_m2:.2f} m²　<span style='color:#888;font-size:.85em'>"
                    f"({area_m2/PING_PER_M2:.2f} 坪)</span></span></div>",
                    unsafe_allow_html=True,
                )
            st.divider()
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:8px 12px;"
                f"border-radius:6px;background:#f0f4ff;border:1px solid #1a3f6f'>"
                f"<b>總計</b><b>{total_m2:.2f} m²（約 {total_m2/PING_PER_M2:.2f} 坪）</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("尚未框選任何空間，請在左邊圖面上開始框選。")
else:
    st.info("請先上傳一份平面圖（PDF 或圖片）開始。")

# ─────────────────────────────────────────────
# 空調負載及選機
# ─────────────────────────────────────────────
st.divider()
st.markdown("#### ❄️ 空調負載及選機")

indoor_models, equip_lookup = load_equipment_data()
if not indoor_models:
    st.caption("⚠️ 尚未連上設備資料表（Google Sheets），室內機下拉選單目前是空的。"
               "需要在 Streamlit Cloud 的 Secrets 加入 `gcp_service_account` 服務帳號設定才能抓到真實機型清單。")

DERIVED_KEYS = ["需求冷房能力", "類型", "室內機冷房能力", "室外機", "連結率"]

def compute_derived(row):
    area = row.get("面積(m²)", 0) or 0
    load = row.get("每坪建議負荷值", 800) or 800
    demand = round(area / 3.3 * load) if area else 0
    indoor = row.get("室內機", "")
    info = equip_lookup.get(indoor, {})
    equip_type = info.get("類型", "")
    return {
        "需求冷房能力": demand,
        "類型": equip_type,
        "室內機冷房能力": info.get("室內機冷房能力", ""),
        "室外機": info.get("室外機", ""),
        "連結率": row.get("連結率", "") if "VRV" in equip_type.upper() else "",
    }

def on_equip_edit():
    """只有在儲存格真的『編輯完成』時，Streamlit 才會觸發這個 callback，
    這裡才去重算自動欄位——不是每次頁面重新整理都重算，才不會干擾正在操作中的下拉選單。"""
    diff = st.session_state.get("equip_data_editor", {})
    table = st.session_state["equip_table"]

    for idx_str, changes in diff.get("edited_rows", {}).items():
        idx = int(idx_str)
        if idx < len(table):
            table[idx].update(changes)
            table[idx].update(compute_derived(table[idx]))

    for new_row in diff.get("added_rows", []):
        row = {
            "編號": f"#{len(table)+1}", "空間名稱": new_row.get("空間名稱", ""),
            "面積(m²)": new_row.get("面積(m²)", 0.0),
            "每坪建議負荷值": new_row.get("每坪建議負荷值", 800),
            "室內機": new_row.get("室內機", ""),
        }
        row.update(compute_derived(row))
        table.append(row)

    for idx in sorted(diff.get("deleted_rows", []), reverse=True):
        if idx < len(table):
            table.pop(idx)

    st.session_state["equip_table"] = table

# 用目前框選到的空間，依「編號」帶入表格；空間名稱可自由改，不會因為重新框選就被蓋掉
if shapes_for_table:
    existing = {row.get("編號"): row for row in (st.session_state["equip_table"] or [])}
    rows = []
    for s in shapes_for_table:
        prev = existing.get(s["name"], {})
        base = {
            "編號": s["name"],
            "空間名稱": prev.get("空間名稱", ""),
            "面積(m²)": s["area"],
            "每坪建議負荷值": prev.get("每坪建議負荷值", 800),
            "室內機": prev.get("室內機", ""),
        }
        base.update({k: prev.get(k, "") for k in DERIVED_KEYS} if prev else compute_derived(base))
        rows.append(base)
    st.session_state["equip_table"] = rows

df_source = st.session_state["equip_table"] or [
    {"編號": "#1", "空間名稱": "", "面積(m²)": 0.0, "每坪建議負荷值": 800, "室內機": "",
     "需求冷房能力": 0, "類型": "", "室內機冷房能力": "", "室外機": "", "連結率": ""}
]
df = pd.DataFrame(df_source)

st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "編號": st.column_config.TextColumn("編號", disabled=True),
        "空間名稱": st.column_config.TextColumn("空間名稱"),
        "面積(m²)": st.column_config.NumberColumn("面積(m²)", min_value=0.0, step=0.1, format="%.2f"),
        "每坪建議負荷值": st.column_config.SelectboxColumn("每坪建議負荷值", options=LOAD_OPTIONS, required=True),
        "需求冷房能力": st.column_config.NumberColumn("需求冷房能力", disabled=True,
                                                     help="= 面積(m²) ÷ 3.3 × 每坪建議負荷值"),
        "室內機": st.column_config.SelectboxColumn("室內機", options=indoor_models or [""]),
        "類型": st.column_config.TextColumn("類型", disabled=True, help="依選定的室內機自動帶出"),
        "室內機冷房能力": st.column_config.TextColumn("室內機冷房能力", disabled=True),
        "室外機": st.column_config.TextColumn("室外機", disabled=True, help="依選定的室內機自動帶出"),
        "連結率": st.column_config.TextColumn("連結率", help="僅 VRV 系列需要填寫"),
    },
    key="equip_data_editor",
    on_change=on_equip_edit,
)
