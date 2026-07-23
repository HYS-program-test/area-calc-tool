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

RENDER_DPI = 144
MAX_CANVAS_WIDTH = 1150
PING_PER_M2 = 3.3058
CROP_PADDING = 25

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {"last_file_key": None, "finished_shapes": [], "claude_review": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def reset_drawing_state():
    st.session_state["finished_shapes"] = []
    st.session_state["claude_review"] = None

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
# Fabric.js 畫布：所有互動（畫矩形/多邊形、選取、拖曳、拉伸、改色、刪除）
# 全部在畫布內部用自己的工具列處理，完全不觸發 Streamlit 重新整理，
# 只有資料寫進 localStorage，等使用者按下方「計算面積」才讀取一次。
# ─────────────────────────────────────────────
CANVAS_HTML = """
<style>
  .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:8px;
             background:#f7f9fc; border:1px solid #e1e8f0; border-radius:8px 8px 0 0; font-family:sans-serif; }
  .toolbar button { padding:6px 12px; border-radius:6px; border:1px solid #ccd6e4; background:#fff;
                     cursor:pointer; font-size:13px; }
  .toolbar button.active { background:#1a3f6f; color:#fff; border-color:#1a3f6f; }
  .toolbar button:hover { filter:brightness(0.97); }
  .toolbar input[type=color] { width:34px; height:30px; border:1px solid #ccd6e4; border-radius:6px; cursor:pointer; }
  .toolbar .sep { width:1px; height:22px; background:#ddd; margin:0 4px; }
  .canvas-wrap { border:1px solid #e1e8f0; border-top:none; border-radius:0 0 8px 8px; overflow:auto; max-height:760px; }
</style>
<div class="toolbar">
  <button id="btn-rect" class="active">▭ 矩形</button>
  <button id="btn-poly">⬠ 多邊形</button>
  <button id="btn-select">↖ 選取／調整</button>
  <div class="sep"></div>
  <input type="color" id="color-picker" value="#FF6347">
  <button id="btn-recolor">套用顏色到選取</button>
  <div class="sep"></div>
  <button id="btn-delete">🗑 刪除選取</button>
  <button id="btn-clear">清空全部</button>
</div>
<div class="canvas-wrap"><canvas id="fabric-canvas"></canvas></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.0/fabric.min.js"></script>
<script>
(function() {
  const bgDataUrl = "__BG_DATA_URL__";
  const initialShapes = __INITIAL_SHAPES__;

  const canvas = new fabric.Canvas('fabric-canvas', {selection: true});
  let mode = 'rect';
  let color = '#FF6347';

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  fabric.Image.fromURL(bgDataUrl, function(img) {
    canvas.setWidth(img.width);
    canvas.setHeight(img.height);
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
    initialShapes.forEach(s => addShapeFromPoints(s.points, s.color, s.type || 'polygon'));
    canvas.renderAll();
    saveState();
  });

  function addShapeFromPoints(points, shapeColor, shapeType) {
    let obj;
    if (shapeType === 'rect') {
      const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
      obj = new fabric.Rect({
        left: Math.min(...xs), top: Math.min(...ys),
        width: Math.max(...xs)-Math.min(...xs), height: Math.max(...ys)-Math.min(...ys),
        fill: hexToRgba(shapeColor, 0.25), stroke: shapeColor, strokeWidth: 3,
        transparentCorners: false, cornerColor: shapeColor, cornerSize: 9,
      });
    } else {
      obj = new fabric.Polygon(points.map(p=>({x:p[0], y:p[1]})), {
        fill: hexToRgba(shapeColor, 0.25), stroke: shapeColor, strokeWidth: 3,
        transparentCorners: false, cornerColor: shapeColor, cornerSize: 9, objectCaching: false,
      });
    }
    canvas.add(obj);
    return obj;
  }

  function setMode(newMode) {
    mode = newMode;
    canvas.selection = (mode === 'select');
    canvas.forEachObject(o => { o.selectable = (mode === 'select'); o.evented = (mode === 'select'); });
    canvas.discardActiveObject();
    document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
    if (mode === 'rect') document.getElementById('btn-rect').classList.add('active');
    if (mode === 'polygon') document.getElementById('btn-poly').classList.add('active');
    if (mode === 'select') document.getElementById('btn-select').classList.add('active');
    canvas.renderAll();
  }

  document.getElementById('btn-rect').onclick = () => setMode('rect');
  document.getElementById('btn-poly').onclick = () => setMode('polygon');
  document.getElementById('btn-select').onclick = () => setMode('select');
  document.getElementById('color-picker').oninput = (e) => { color = e.target.value; };
  document.getElementById('btn-recolor').onclick = () => {
    const active = canvas.getActiveObject();
    if (active) {
      active.set({stroke: color, fill: hexToRgba(color, 0.25)});
      canvas.renderAll();
      saveState();
    }
  };
  document.getElementById('btn-delete').onclick = () => {
    const active = canvas.getActiveObject();
    if (active) { canvas.remove(active); saveState(); }
  };
  document.getElementById('btn-clear').onclick = () => {
    canvas.getObjects().slice().forEach(o => canvas.remove(o));
    saveState();
  };

  let isDrawingRect = false, rectStart = null, activeRect = null;
  let polyPoints = [], polyMarkers = [], polyLine = null;

  canvas.on('mouse:down', function(o) {
    if (mode === 'rect') {
      isDrawingRect = true;
      const p = canvas.getPointer(o.e);
      rectStart = {x: p.x, y: p.y};
      activeRect = new fabric.Rect({
        left: p.x, top: p.y, width: 0, height: 0,
        fill: hexToRgba(color, 0.25), stroke: color, strokeWidth: 3, selectable: false, evented: false,
      });
      canvas.add(activeRect);
    } else if (mode === 'polygon') {
      const p = canvas.getPointer(o.e);
      if (polyPoints.length > 2) {
        const dx = p.x - polyPoints[0].x, dy = p.y - polyPoints[0].y;
        if (Math.sqrt(dx*dx + dy*dy) < 16) { finishPolygon(); return; }
      }
      polyPoints.push({x: p.x, y: p.y});
      const marker = new fabric.Circle({left:p.x-5, top:p.y-5, radius:5, fill:color, selectable:false, evented:false});
      canvas.add(marker); polyMarkers.push(marker);
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
      } else if (activeRect) {
        activeRect.set({selectable: (mode==='select'), evented: (mode==='select')});
        saveState();
      }
      activeRect = null;
    }
  });

  function updatePolyLine() {
    if (polyLine) canvas.remove(polyLine);
    if (polyPoints.length > 1) {
      polyLine = new fabric.Polyline(polyPoints, {fill:'', stroke:color, strokeWidth:2, selectable:false, evented:false});
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
        transparentCorners: false, cornerColor: color, cornerSize: 9, objectCaching: false,
        selectable: (mode==='select'), evented: (mode==='select'),
      });
      canvas.add(poly);
      saveState();
    }
    polyPoints = []; polyMarkers = []; polyLine = null;
  }

  document.addEventListener('keydown', function(e) {
    if ((e.key === 'Delete' || e.key === 'Backspace') && mode === 'select') {
      const active = canvas.getActiveObject();
      if (active) { canvas.remove(active); saveState(); }
    }
  });

  canvas.on('object:modified', saveState);

  function saveState() {
    const shapes = canvas.getObjects().filter(o => o.type === 'rect' || o.type === 'polygon').map(o => {
      const matrix = o.calcTransformMatrix();
      let pts;
      if (o.type === 'rect') {
        const w = o.width * o.scaleX, h = o.height * o.scaleY;
        pts = [{x:-w/2,y:-h/2},{x:w/2,y:-h/2},{x:w/2,y:h/2},{x:-w/2,y:h/2}].map(c => {
          const t = fabric.util.transformPoint(c, matrix); return [t.x, t.y];
        });
      } else {
        pts = o.points.map(p => {
          const t = fabric.util.transformPoint({x:p.x-o.pathOffset.x, y:p.y-o.pathOffset.y}, matrix);
          return [t.x, t.y];
        });
      }
      return {points: pts, color: o.stroke, type: o.type};
    });
    window.localStorage.setItem('area_calc_shapes_v2', JSON.stringify(shapes));
  }

  setMode('rect');
})();
</script>
"""

def render_canvas(bg_img: Image.Image, shapes: list, height: int):
    buf = io.BytesIO()
    bg_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    html = CANVAS_HTML.replace("__BG_DATA_URL__", f"data:image/jpeg;base64,{b64}")
    html = html.replace("__INITIAL_SHAPES__", json.dumps(shapes))
    components.html(html, height=height + 70, scrolling=True)

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

    sc1, sc2, sc3 = st.columns([1.3, 1, 1])
    with sc1:
        scale_ratio = st.number_input(
            f"比例尺 1:N｜{'✅自動偵測' if auto_scale else '⚠️請手動輸入'}",
            min_value=1, value=auto_scale or 100, step=10,
        )
    with sc2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        calc_clicked = st.button("📐 計算面積", type="primary", use_container_width=True,
                                  help="把畫布上目前的框選結果讀進來計算面積")
    with sc3:
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

    st.caption("畫布左上角是自己的工具列：矩形／多邊形／選取（拖曳移動、拉角縮放、Delete鍵刪除）／改色／清空，都在畫布內操作，不會整頁重新整理。畫完按上面「📐 計算面積」才會把結果算出來。")
    render_canvas(disp_img, st.session_state["finished_shapes"], disp_img.height)

    if calc_clicked:
        raw = st_javascript("await new Promise(r => r(window.localStorage.getItem('area_calc_shapes_v2')));")
        if raw and raw != 0:
            try:
                st.session_state["finished_shapes"] = json.loads(raw)
            except Exception:
                st.warning("讀取失敗，請再按一次「計算面積」")
        st.rerun()

    # ── 結果 ──────────────────────────
    if st.session_state["finished_shapes"]:
        total_m2 = sum(polygon_area_px2(s["points"]) * m2_per_px2_display for s in st.session_state["finished_shapes"])
        res_cols = st.columns(min(len(st.session_state["finished_shapes"]), 8) + 1)
        for i, shape in enumerate(st.session_state["finished_shapes"][:8]):
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
                for i, s in enumerate(st.session_state["finished_shapes"])
            ]
            if st.button("請 Claude 協助核對"):
                overlay = np.array(disp_img).copy()
                for i, shape in enumerate(st.session_state["finished_shapes"]):
                    pts_np = np.array(shape["points"], dtype=np.int32)
                    hexc = shape["color"].lstrip("#")
                    r, g, b = int(hexc[0:2],16), int(hexc[2:4],16), int(hexc[4:6],16)
                    cv2.polylines(overlay, [pts_np], True, (r, g, b), 3)
                    cx, cy = int(np.mean(pts_np[:, 0])), int(np.mean(pts_np[:, 1]))
                    cv2.putText(overlay, f"#{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                with st.spinner("Claude 正在對照圖面檢查中…"):
                    review_text = ask_claude_review(overlay, results)
                st.session_state["claude_review"] = review_text
            if st.session_state.get("claude_review"):
                st.info(st.session_state["claude_review"])
    else:
        st.caption("尚未框選任何空間，請在上方畫布開始框選，完成後按「計算面積」。")
else:
    st.info("請先上傳一份平面圖（PDF 或圖片）開始。")
