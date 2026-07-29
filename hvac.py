from __future__ import annotations

M2_PER_PING = 3.305785
W_TO_KCAL_PER_HOUR = 0.859845


def polygon_area_pixels(points: list[dict]) -> float:
    if len(points) < 3:
        return 0.0

    total = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        total += p1["x"] * p2["y"] - p2["x"] * p1["y"]

    return abs(total) / 2


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_rows(
    rooms: list[dict],
    image_width: int,
    image_height: int,
    pixels_per_m2: float,
):
    """建立空調負荷與設備選型結果。

    每坪建議負荷值的單位為 kcal/h/坪。
    總熱負荷同時換算為 W，方便後續與設備冷房能力比對。
    """
    rows = []

    for room in rooms:
        px_points = [
            {
                "x": p["x"] / 1000 * image_width,
                "y": p["y"] / 1000 * image_height,
            }
            for p in room["points"]
        ]

        area_m2 = polygon_area_pixels(px_points) / pixels_per_m2
        area_ping = area_m2 / M2_PER_PING

        # 新資料優先使用 per_ping_load。
        # 舊資料若只有 unit_load(W/m²)，則自動換算成 kcal/h/坪。
        if room.get("per_ping_load") not in (None, ""):
            per_ping_load = _to_float(room.get("per_ping_load"), 0.0)
        else:
            unit_load_w_m2 = _to_float(room.get("unit_load"), 120.0)
            per_ping_load = (
                unit_load_w_m2 * M2_PER_PING * W_TO_KCAL_PER_HOUR
            )

        total_heat_kcal_h = area_ping * per_ping_load
        total_heat_w = total_heat_kcal_h / W_TO_KCAL_PER_HOUR

        indoor_model = room.get("indoor_model") or "-"
        indoor_capacity_kw = _to_float(
            room.get("indoor_capacity_kw"),
            0.0,
        )
        outdoor_model = room.get("outdoor_model") or "-"
        outdoor_capacity_kw = _to_float(
            room.get("outdoor_capacity_kw"),
            0.0,
        )

        connection_rate = (
            indoor_capacity_kw / outdoor_capacity_kw * 100
            if outdoor_capacity_kw > 0
            else None
        )

        rows.append(
            {
                "編號": room["id"],
                "區域名稱": room["name"],
                "面積 (m²)": round(area_m2, 2),
                "空間類型": room.get("room_type", "一般辦公室"),
                "每坪建議負荷值 (kcal/h/坪)": round(per_ping_load),
                "總熱負荷 (W)": round(total_heat_w),
                "室內機型號": indoor_model,
                "室內機冷房能力 (kW)": (
                    round(indoor_capacity_kw, 2)
                    if indoor_capacity_kw > 0
                    else "-"
                ),
                "室外機型號": outdoor_model,
                "連結率 (%)": (
                    round(connection_rate, 1)
                    if connection_rate is not None
                    else "-"
                ),
            }
        )

    return rows
