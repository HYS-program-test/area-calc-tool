import streamlit as st
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript
import re
import io
import base64
import json
import anthropic

st.set_page_config(page_title="平面圖面積計算工具", page_icon="📐", layout="wide")

st.title("📐 平面圖面積計算工具")
st.caption("上傳平面圖 → 矩形即時拖曳 或 多邊形點角點（點回起點自動封閉）→ 框框可個別拖曳／拉伸／刪除")

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
RENDER_DPI = 144
MAX_CANVAS_WIDTH = 1000
PING_PER_M2 = 3.3058
DEFAULT_COLOR = "#FF6347"
CROP_PADDING = 25

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "last_file_key": None,
        "finished_shapes": [],   # [{"points":[[x,y],...], "color":"#RRGGBB"}, ...]
        "claude_review": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def reset_drawing_state():
    st.session_state["finished_shapes"] = []
    st.session_state["claude_review"] = None

# ─────────────────────────────────────────────
# PDF / 圖片載入與裁切（皆快取，避免每次互動都重新運算造成卡頓）
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
    """自動裁切掉圖面四周空白／外框，讓建築本體置中放大。
    用「墨跡密度」而非單純外框來判斷，避免滿版的圖框線（密度低、但bbox很大）
    把裁切範圍撐成整頁。"""
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
        density = area / bbox_area
        score = density * area  # 密度 x 面積：排除細長外框，也排除太小的雜訊
        if score > best_score:
            best_score = score
            best_idx = idx
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
    """Shoelace 公式，支援矩形與任意不規則多邊形"""
    n = len(pts)
    area = 0.0
    for j in range(n):
        x1, y1 = pts[j]
        x2, y2 = pts[(j + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def ask_claude_review(overlay_img: np.ndarray, results: list) -> str:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ 尚未設定 ANTHROPIC_API_KEY（請至 Streamlit Cloud → Settings → Secrets 加入）"
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
# Fabric.js 互動畫布（透過 CDN 載入，不需要額外的伺服器靜態檔案）
# ─────────────────────────────────────────────
FABRIC_CANVAS_HTML = """
<div style="border:1px solid #ddd;border-radius:8px;overflow:auto;max-height:800px;">
  <canvas id="fabric-canvas"></canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.0/fabric.min.js"></script>
<script>
(function() {
  const bgDataUrl = "__BG_DATA_URL__";
  const initialShapes = __INITIAL_SHAPES__;
  const mode = "__MODE__";
  const color = "__COLOR__";

  const canvasEl = document.getElementById('fabric-canvas');
  const canvas = new fabric.Canvas(canvasEl, {selection: mode === 'select'});

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  fabric.Image.fromURL(bgDataUrl, function(img) {
    canvas.setWidth(img.width);
    canvas.setHeight(img.height);
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));

    initialShapes.forEach(function(shape) {
      addShapeFromPoints(shape.points, shape.color, shape.type || 'polygon');
    });
    canvas.renderAll();
  });

  function addShapeFromPoints(points, shapeColor, shapeType) {
    if (shapeType === 'rect') {
      const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
      const rect = new fabric.Rect({
        left: Math.min(...xs), top: Math.min(...ys),
        width: Math.max(...xs)-Math.min(...xs), height: Math.max(...ys)-Math.min(...ys),
        fill: hexToRgba(shapeColor, 0.25), stroke: shapeColor, strokeWidth: 3,
        transparentCorners: false, cornerColor: shapeColor, cornerSize: 9,
      });
      canvas.add(rect);
    } else {
      const poly = new fabric.Polygon(points.map(p=>({x:p[0], y:p[1]})), {
        fill: hexToRgba(shapeColor, 0.25), stroke: shapeColor, strokeWidth: 3,
        transparentCorners: false, cornerColor: shapeColor, cornerSize: 9,
        objectCaching: false,
      });
      canvas.add(poly);
    }
  }

  let isDrawingRect = false, rectStart = null, activeRect = null;
  let polyPoints = [], polyMarkers = [], polyLine = null;

  canvas.on('mouse:down', function(o) {
    if (mode === 'rect') {
      if (canvas.getActiveObject()) return;
      isDrawingRect = true;
      const p = canvas.getPointer(o.e);
      rectStart = {x: p.x, y: p.y};
      activeRect = new fabric.Rect({
        left: p.x, top: p.y, width: 0, height: 0,
        fill: hexToRgba(color, 0.25), stroke: color, strokeWidth: 3,
        transparentCorners: false, cornerColor: color, cornerSize: 9,
      });
      canvas.add(activeRect);
    } else if (mode === 'polygon') {
      if (canvas.getActiveObject()) return;
      const p = canvas.getPointer(o.e);
      if (polyPoints.length > 2) {
        const dx = p.x - polyPoints[0].x, dy = p.y - polyPoints[0].y;
        if (Math.sqrt(dx*dx + dy*dy) < 16) {
          finishPolygon();
          return;
        }
      }
      polyPoints.push({x: p.x, y: p.y});
      const marker = new fabric.Circle({
        left: p.x - 5, top: p.y - 5, radius: 5, fill: color, selectable: false, evented: false,
      });
      canvas.add(marker);
      polyMarkers.push(marker);
      updatePolyLine();
    }
  });

  canvas.on('mouse:move', function(o) {
    if (!isDrawingRect || !activeRect) return;
    const p = canvas.getPointer(o.e);
    activeRect.set({
      left: Math.min(p.x, rectStart.x), top: Math.min(p.y, rectStart.y),
      width: Math.abs(p.x - rectStart.x), height: Math.abs(p.y - rectStart.y),
    });
    canvas.requestRenderAll();
  });

  canvas.on('mouse:up', function() {
    if (isDrawingRect) {
      isDrawingRect = false;
      if (activeRect && (activeRect.width < 5 || activeRect.height < 5)) {
        canvas.remove(activeRect);
      } else {
        saveState();
      }
      activeRect = null;
    }
  });

  function updatePolyLine() {
    if (polyLine) canvas.remove(polyLine);
    if (polyPoints.length > 1) {
      polyLine = new fabric.Polyline(polyPoints, {
        fill: '', stroke: color, strokeWidth: 2, selectable: false, evented: false,
      });
      canvas.add(polyLine);
    }
    canvas.renderAll();
  }

  function finishPolygon() {
    polyMarkers.forEach(m => canvas.remove(m));
    if (polyLine) canvas.remove(polyLine);
    if (polyPoints.length >= 3) {
      const poly = new fabric.Polygon(polyPoints, {
        fill: hexToRgba(color, 0.25), stroke: color, strokeWidth: 3,
        transparentCorners: false, cornerColor: color, cornerSize: 9,
        objectCaching: false,
      });
      canvas.add(poly);
      saveState();
    }
    polyPoints = []; polyMarkers = []; polyLine = null;
  }

  document.addEventListener('keydown', function(e) {
    if ((e.key === 'Delete' || e.key === 'Backspace')) {
      const active = canvas.getActiveObject();
      if (active && active.selectable) {
        canvas.remove(active);
        saveState();
      }
    }
  });

  canvas.on('object:modified', saveState);

  function saveState() {
    const shapes = canvas.getObjects().filter(o => o.type === 'rect' || o.type === 'polygon').map(o => {
      const matrix = o.calcTransformMatrix();
      let pts;
      if (o.type === 'rect') {
        const w = o.width * o.scaleX, h = o.height * o.scaleY;
        const corners = [{x:-w/2,y:-h/2},{x:w/2,y:-h/2},{x:w/2,y:h/2},{x:-w/2,y:h/2}];
        pts = corners.map(c => {
          const t = fabric.util.transformPoint(c, matrix);
          return [t.x, t.y];
        });
      } else {
        pts = o.points.map(p => {
          const t = fabric.util.transformPoint({x: p.x - o.pathOffset.x, y: p.y - o.pathOffset.y}, matrix);
          return [t.x, t.y];
        });
      }
      return {points: pts, color: o.stroke, type: o.type};
    });
    window.localStorage.setItem('area_calc_shapes_v1', JSON.stringify(shapes));
  }

  saveState();
})();
</script>
"""

def render_canvas(bg_img: Image.Image, shapes: list, mode: str, color: str, height: int):
    buf = io.BytesIO()
    bg_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    bg_data_url = f"data:image/jpeg;base64,{b64}"

    html = FABRIC_CANVAS_HTML
    html = html.replace("__BG_DATA_URL__", bg_data_url)
    html = html.replace("__INITIAL_SHAPES__", json.dumps(shapes))
    html = html.replace("__MODE__", mode)
    html = html.replace("__COLOR__", color)
    components.html(html, height=height + 20, scrolling=True)

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
uploaded = st.file_uploader("上傳平面圖（PDF 或圖片）", type=["pdf", "png", "jpg", "jpeg"])

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

    col_scale1, col_scale2 = st.columns([1, 2])
    with col_scale1:
        if auto_scale:
            st.success(f"✅ 自動偵測到比例尺 1:{auto_scale}（可在右方修正）")
        elif is_pdf:
            st.warning("⚠️ 未在圖面文字中偵測到比例尺，請手動輸入")
        else:
            st.warning("⚠️ 圖片檔無法自動偵測比例尺，請手動輸入")
    with col_scale2:
        scale_ratio = st.number_input(
            "比例尺（輸入 1:N 裡的 N）", min_value=1, value=auto_scale or 100, step=10,
        )

    disp_img, display_scale = resize_display_cached(img_cropped, file_key, MAX_CANVAS_WIDTH)

    m_per_px_at_render = (2.54 / RENDER_DPI / 100) * scale_ratio if is_pdf else None
    if m_per_px_at_render:
        m_per_px_display = m_per_px_at_render / display_scale
    else:
        m_per_px_display = (2.54 / 96 / 100) * scale_ratio / display_scale
    m2_per_px2_display = m_per_px_display ** 2

    # ── 工具列 ──────────────────────────
    tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([1.6, 1, 1, 1.3])
    with tool_col1:
        draw_mode = st.radio("框選模式", ["矩形", "多邊形", "選取／調整"], horizontal=True,
                              help="矩形：拖曳即時畫出矩形。多邊形：依序點角點，點回起點附近自動封閉。選取／調整：拖曳移動、拉角點縮放，按 Delete 刪除選取的框。")
    with tool_col2:
        shape_color_hex = st.color_picker("新框的顏色", DEFAULT_COLOR)
    with tool_col3:
        if st.button("🔄 同步目前框選狀態", use_container_width=True,
                      help="在畫布上畫完/調整完之後，按這裡把最新結果帶回來計算面積"):
            raw = st_javascript("await new Promise(r => r(window.localStorage.getItem('area_calc_shapes_v1')));")
            if raw and raw != 0:
                try:
                    st.session_state["finished_shapes"] = json.loads(raw)
                except Exception:
                    st.warning("同步失敗，請再試一次")
            st.rerun()
    with tool_col4:
        if st.button("🗑️ 清空全部重來", use_container_width=True):
            reset_drawing_state()
            st.rerun()

    mode_map = {"矩形": "rect", "多邊形": "polygon", "選取／調整": "select"}
    render_canvas(disp_img, st.session_state["finished_shapes"], mode_map[draw_mode], shape_color_hex, disp_img.height)

    st.caption("畫完或調整完，記得按上面「🔄 同步目前框選狀態」，面積才會更新。")

    # ── 已封閉空間清單 ──────────────────────────
    if st.session_state["finished_shapes"]:
        st.markdown("**已框選的空間：**")
        total_m2 = 0.0
        for i, shape in enumerate(st.session_state["finished_shapes"]):
            area_m2 = polygon_area_px2(shape["points"]) * m2_per_px2_display
            total_m2 += area_m2
            c1, c2 = st.columns([0.6, 5.4])
            with c1:
                st.markdown(
                    f"<div style='width:20px;height:20px;border-radius:4px;background:{shape['color']};margin-top:6px'></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.write(f"#{i+1}　約 {area_m2:.2f} m²（{len(shape['points'])} 個角點）")

        st.markdown(f"### 總計：{total_m2:.2f} m²（約 {total_m2/PING_PER_M2:.2f} 坪）")

        results = [
            {"id": i + 1, "area_m2": polygon_area_px2(s["points"]) * m2_per_px2_display, "points": s["points"]}
            for i, s in enumerate(st.session_state["finished_shapes"])
        ]

        st.markdown("**🤖 Claude 輔助核對**：對照原圖，幫忙標註每個框對應的空間、指出可疑或漏框的地方（僅供參考，不影響面積數字）")
        if st.button("🤖 請 Claude 協助核對", use_container_width=True):
            overlay = np.array(disp_img).copy()
            for i, shape in enumerate(st.session_state["finished_shapes"]):
                pts_np = np.array(shape["points"], dtype=np.int32)
                b = int(shape["color"][1:3], 16); g = int(shape["color"][3:5], 16); r = int(shape["color"][5:7], 16)
                cv2.polylines(overlay, [pts_np], True, (r, g, b), 3)
                cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))
                cv2.putText(overlay, f"#{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
            with st.spinner("Claude 正在對照圖面檢查中…"):
                review_text = ask_claude_review(overlay, results)
            st.session_state["claude_review"] = review_text

        if st.session_state.get("claude_review"):
            st.info(st.session_state["claude_review"])
else:
    st.info("請先上傳一份平面圖（PDF 或圖片）開始。")
