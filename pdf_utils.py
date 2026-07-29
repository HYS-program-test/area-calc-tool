import io
import fitz
from PIL import Image

def _drawing_color(item):
    color = item.get("color")
    if not color:
        return None
    return tuple(round(float(v), 3) for v in color)

def detect_building_bbox(page: fitz.Page):
    blue_rects = []
    for item in page.get_drawings():
        color = _drawing_color(item)
        rect = item["rect"]
        if (
            color == (0.0, 0.0, 1.0)
            and 5 <= rect.width <= 80
            and 8 <= rect.height <= 100
        ):
            blue_rects.append(rect)

    if len(blue_rects) >= 4:
        x0 = min(r.x0 for r in blue_rects)
        y0 = min(r.y0 for r in blue_rects)
        x1 = max(r.x1 for r in blue_rects)
        y1 = max(r.y1 for r in blue_rects)
        bbox = fitz.Rect(x0 - 30, y0 - 30, x1 + 45, y1 + 75) & page.rect
        return bbox, {"method": "blue_structural_columns", "column_count": len(blue_rects)}

    central = fitz.Rect(
        page.rect.width * 0.10,
        page.rect.height * 0.10,
        page.rect.width * 0.90,
        page.rect.height * 0.90,
    )
    candidates = []
    for item in page.get_drawings():
        rect = item["rect"]
        if not rect.intersects(central):
            continue
        if rect.width < 2 and rect.height < 2:
            continue
        if rect.width > page.rect.width * 0.85 and rect.height > page.rect.height * 0.85:
            continue
        candidates.append(rect)

    if not candidates:
        return page.rect, {"method": "full_page_fallback", "column_count": len(blue_rects)}

    x0 = min(r.x0 for r in candidates)
    y0 = min(r.y0 for r in candidates)
    x1 = max(r.x1 for r in candidates)
    y1 = max(r.y1 for r in candidates)
    bbox = fitz.Rect(x0 - 20, y0 - 20, x1 + 20, y1 + 20) & page.rect
    return bbox, {"method": "central_vector_fallback", "column_count": len(blue_rects)}

def render_pdf_page(pdf_bytes: bytes, page_index: int = 0, dpi: int = 220):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            raise ValueError("PDF 沒有可讀取頁面")
        page = doc[page_index]
        bbox, meta = detect_building_bbox(page)
        scale = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=bbox, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        meta.update({
            "page_count": doc.page_count,
            "page_index": page_index,
            "crop_box_points": [round(bbox.x0,2), round(bbox.y0,2), round(bbox.x1,2), round(bbox.y1,2)],
            "output_size_pixels": list(image.size),
        })
        return image, meta
    finally:
        doc.close()
