from __future__ import annotations

from typing import Any, Dict, List

M2_PER_PING = 3.305785


def polygon_area_pixels(points: List[Dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0

    total = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        total += p1["x"] * p2["y"] - p2["x"] * p1["y"]

    return abs(total) / 2


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_rows(
    rooms: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    pixels_per_m2: float,
) -> List[Dict[str, Any]]:
    """建立空調負荷計算表。

    面積(坪) = 面積(m²) / 3.305785
    總熱負荷(kcal/h) = 面積(坪) × 每坪建議負荷值(kcal/h/坪)
    """
    rows: List[Dict[str, Any]] = []

    for room in rooms:
        px_points = [
            {
                "x": p["x"] / 1000 * image_width,
                "y": p["y"] / 1000 * image_height,
            }
            for p in room.get("points", [])
        ]

        area_m2 = polygon_area_pixels(px_points) / pixels_per_m2
        area_ping = area_m2 / M2_PER_PING
        per_ping_load = _to_float(room.get("per_ping_load"), 650.0)
        total_heat_kcal_h = area_ping * per_ping_load

        rows.append(
            {
                "編號": room.get("id"),
                "區域名稱": room.get("name", ""),
                "面積 (m²)": round(area_m2, 2),
                "面積 (坪)": round(area_ping, 2),
                "空間類型": room.get("room_type", "一般辦公室"),
                "每坪建議負荷值 (kcal/h/坪)": round(per_ping_load, 2),
                "總熱負荷 (kcal/h)": round(total_heat_kcal_h, 2),
                "室內機型號": room.get("indoor_model") or "",
                "室內機冷房能力 (kW)": (
                    _to_float(room.get("indoor_capacity_kw"))
                    if room.get("indoor_capacity_kw") not in (None, "", "-")
                    else None
                ),
                "室外機型號": room.get("outdoor_model") or "",
                "連結率 (%)": (
                    _to_float(room.get("connection_rate"))
                    if room.get("connection_rate") not in (None, "", "-")
                    else None
                ),
            }
        )

    return rows
