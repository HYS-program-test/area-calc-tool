from PIL import Image, ImageDraw

def norm_to_pixel(points, width, height):
    return [
        (
            round(p.x / 1000 * (width - 1)),
            round(p.y / 1000 * (height - 1)),
        )
        for p in points
    ]

def polygon_area_pixels(points):
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2

def draw_result(image: Image.Image, result):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    areas = []

    for room in result.rooms:
        pts = norm_to_pixel(room.points, overlay.width, overlay.height)
        if len(pts) < 3:
            continue

        draw.polygon(pts, fill=(255, 0, 0, 28))
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=5)
        draw.text(pts[0], room.name, fill=(0, 0, 0, 255))

        areas.append({
            "name": room.name,
            "room_type": room.room_type,
            "confidence": room.confidence,
            "area_pixels": polygon_area_pixels(pts),
            "points_pixels": pts,
        })

    return overlay, areas
