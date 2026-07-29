from __future__ import annotations

def polygon_area_pixels(points: list[dict]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        total += p1["x"] * p2["y"] - p2["x"] * p1["y"]
    return abs(total) / 2

def calculate_rows(rooms: list[dict], image_width: int, image_height: int, pixels_per_m2: float):
    rows = []
    for room in rooms:
        px_points = [
            {"x": p["x"] / 1000 * image_width, "y": p["y"] / 1000 * image_height}
            for p in room["points"]
        ]
        area_m2 = polygon_area_pixels(px_points) / pixels_per_m2
        sensible = area_m2 * float(room.get("unit_load", 120))
        latent = sensible * 0.55
        total = sensible + latent
        rt = total / 3517
        rows.append({
            "編號": room["id"],
            "區域名稱": room["name"],
            "面積 (m²)": round(area_m2, 2),
            "空間類型": room.get("room_type", "一般辦公室"),
            "單位負荷 (W/m²)": room.get("unit_load", 120),
            "顯熱負荷 (W)": round(sensible),
            "潛熱負荷 (W)": round(latent),
            "總冷負荷 (W)": round(total),
            "建議空調能力 (RT)": round(rt, 2),
            "納入計算": room.get("included", True),
        })
    return rows
