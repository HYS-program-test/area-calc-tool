import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
try:
    from streamlit_image_annotation import detection as st_detection
    from streamlit_image_annotation.Detection import get_colormap as _sia_get_colormap
    HAS_ANNOTATION_PKG = True
except Exception:
    HAS_ANNOTATION_PKG = False
import re
import io
import base64
import anthropic

st.set_page_config(page_title="平面圖面積計算工具", page_icon="📐", layout="wide")

RENDER_DPI = 144
PING_PER_M2 = 3.3058
CROP_PADDING = 25
FIXED_COLORS = ["#FF6347", "#3B82F6", "#22C55E", "#F59E0B", "#A855F7", "#06B6D4"]
COLOR_LABELS = ["1", "2", "3", "4", "5", "6"]  # 純編號；實際顏色仍依 ANNOTATION_REAL_COLORS 的順序正確對應，不受標籤文字影響

# 直接呼叫套件本身的 get_colormap（不是自己重新兜一份），
# 確保跟畫面上矩形工具實際顯示的顏色來源完全一致，不會有重新實作造成的落差。
if HAS_ANNOTATION_PKG:
    _colormap_dict = _sia_get_colormap(COLOR_LABELS, colormap_name="gist_rainbow")
    ANNOTATION_REAL_COLORS = [_colormap_dict[label] for label in COLOR_LABELS]
else:
    ANNOTATION_REAL_COLORS = FIXED_COLORS
LOAD_OPTIONS = list(range(400, 1300, 100))  # 400~1200，每100一個
DEVICE_CATEGORIES = ["RA", "SA", "MA", "VRV"]

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "last_file_key": None,
        "finished_shapes": [],   # [{"points":[(x,y),...], "color":(b,g,r), "group": str|None}]
        "claude_review": None,
        "equip_table": None,
        "group_counter": 0,
        "annot_bboxes": [],
        "annot_labels": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def reset_drawing_state():
    st.session_state["finished_shapes"] = []
    st.session_state["claude_review"] = None
    st.session_state["annot_bboxes"] = []
    st.session_state["annot_labels"] = []

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

def draw_all(base_arr: np.ndarray) -> np.ndarray:
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
    return arr

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

    working_width = 1150
    disp_img, display_scale = resize_display_cached(img_cropped, f"{file_key}_{working_width}", working_width)
    disp_arr_base = np.array(disp_img)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.caption(f"📏 目前工作圖實際像素尺寸：{disp_img.width} × {disp_img.height} px"
                   "（畫面上的框會撐滿版面顯示，所以看起來大小差不多是正常的，這裡的數字才是真的尺寸）")

        sc_col, clr_col = st.columns([3, 1])
        with sc_col:
            scale_ratio = st.number_input(
                f"比例尺 1:N｜{'✅自動' if auto_scale else '⚠️手動'}",
                min_value=1, value=auto_scale or 100, step=10,
            )
        with clr_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ 清空重來", use_container_width=True):
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

        working_arr = draw_all(disp_arr_base)

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

        if HAS_ANNOTATION_PKG:
            with st.expander("🖱️ 矩形工具（可直接拖曳、縮放調整）", expanded=True):
                st.caption("畫矩形前先選一個編號（下面 1~6 對應 6 種顏色），"
                           "畫完可以直接拖曳邊角調整大小、選取後按 Delete 鍵刪除。確認後按「套用」才會"
                           "加進正式的面積結果清單；不規則（L型等）空間可以用多個矩形拼湊，再到右邊清單勾選合併。")
                try:
                    annot_result = st_detection(
                        disp_img, label_list=COLOR_LABELS,
                        bboxes=st.session_state["annot_bboxes"],
                        labels=st.session_state["annot_labels"],
                        height=disp_img.height, width=disp_img.width,
                        key=f"annot_{file_key}",
                    )
                    if annot_result is not None:
                        # 記住目前畫布上的狀態，下次重繪（例如按了套用之後）才不會被清空、消失
                        st.session_state["annot_bboxes"] = [item["bbox"] for item in annot_result]
                        st.session_state["annot_labels"] = [item.get("label_id", 0) for item in annot_result]

                    if annot_result and st.button("✅ 套用這些矩形到面積結果", key="apply_annot_rects"):
                        for item in annot_result:
                            x, y, bw, bh = item["bbox"]
                            pts = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
                            color_hex = ANNOTATION_REAL_COLORS[item.get("label_id", 0) % len(ANNOTATION_REAL_COLORS)]
                            color = hex_to_bgr(color_hex)
                            st.session_state["finished_shapes"].append({"points": pts, "color": color})
                        st.success(f"已套用 {len(annot_result)} 個矩形，畫布上的矩形保留不變，可以繼續調整或再畫新的。")
                except Exception as e:
                    st.error(f"矩形工具載入失敗：{e}")

    # ── 右半部：框選後的面積結果，直向清單，不管左邊工具有沒有摺疊都一直顯示 ──────────
    with right_col:
        st.markdown("##### 📊 面積結果")
        st.caption("要把多個矩形合計成一個不規則空間（例如 L 型），勾選下面對應的幾筆、按「合併勾選項目」。")
        if st.session_state["finished_shapes"]:
            shapes = st.session_state["finished_shapes"]
            # 依 group 分組：group 是 None 的自己單獨一組，其餘同 group 值的合併顯示成一列
            groups = {}
            for i, s in enumerate(shapes):
                gid = s.get("group")
                key = gid if gid is not None else f"solo_{i}"
                groups.setdefault(key, []).append(i)

            total_m2 = 0.0
            delete_indices = []
            selected_for_merge = []
            group_num = 0
            for key, idxs in groups.items():
                group_num += 1
                group_area_m2 = sum(polygon_area_px2(shapes[i]["points"]) * m2_per_px2_display for i in idxs)
                total_m2 += group_area_m2
                b, g, r = shapes[idxs[0]]["color"]
                label = f"#{group_num}" if len(idxs) == 1 else f"#{group_num}（合併 {len(idxs)} 筆）"
                shapes_for_table.append({"name": label, "area": round(group_area_m2, 2)})

                chk_col, row_col, del_col = st.columns([0.6, 4.4, 1])
                with chk_col:
                    checked = st.checkbox("", key=f"merge_chk_{key}", label_visibility="collapsed")
                    if checked:
                        selected_for_merge.append(idxs)
                with row_col:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"padding:8px 12px;margin-bottom:6px;border-radius:6px;"
                        f"background:rgba({r},{g},{b},0.10);border-left:4px solid rgb({r},{g},{b})'>"
                        f"<b style='color:rgb({r},{g},{b})'>{label}</b>"
                        f"<span>{group_area_m2:.2f} m²　<span style='color:#888;font-size:.85em'>"
                        f"({group_area_m2/PING_PER_M2:.2f} 坪)</span></span></div>",
                        unsafe_allow_html=True,
                    )
                with del_col:
                    if st.button("✕", key=f"del_area_shape_{key}", help="刪除這一筆（合併的會整組刪除）"):
                        delete_indices.extend(idxs)

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if st.button("🔗 合併勾選項目", use_container_width=True,
                              disabled=len(selected_for_merge) < 2):
                    st.session_state["group_counter"] += 1
                    new_gid = f"g{st.session_state['group_counter']}"
                    for idxs in selected_for_merge:
                        for i in idxs:
                            shapes[i]["group"] = new_gid
                    st.rerun()
            with m_col2:
                if st.button("✂️ 全部取消合併", use_container_width=True):
                    for s in shapes:
                        s.pop("group", None)
                    st.rerun()

            if delete_indices:
                for i in sorted(set(delete_indices), reverse=True):
                    shapes.pop(i)
                st.rerun()

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
