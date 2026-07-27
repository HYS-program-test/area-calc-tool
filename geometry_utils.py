from __future__ import annotations

import math
from typing import Any, Iterable

from shapely.geometry import Polygon


Point = tuple[float, float]


def polygon_area_px2(points: Iterable[Point]) -> float:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return 0.0

    polygon = Polygon(pts)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    return 0.0 if polygon.is_empty else float(polygon.area)


def pixel_area_to_m2(
    area_px2: float,
    px_per_meter: float | None,
) -> float | None:
    if not px_per_meter or px_per_meter <= 0:
        return None
    return float(area_px2) / float(px_per_meter) ** 2


def px_per_meter_from_line(
    p1: Point,
    p2: Point,
    actual_m: float,
) -> float:
    if actual_m <= 0:
        raise ValueError("實際長度必須大於 0。")

    pixel_length = math.dist(p1, p2)
    if pixel_length <= 0:
        raise ValueError("校正線長度必須大於 0。")

    return pixel_length / actual_m


def cooling_load(
    area_m2: float | None,
    load_per_ping: float,
) -> dict[str, float | None]:
    if area_m2 is None:
        return {
            "ping": None,
            "kcal_h": None,
            "kw": None,
            "btu_h": None,
        }

    ping = area_m2 / 3.3058
    kcal_h = ping * load_per_ping
    return {
        "ping": ping,
        "kcal_h": kcal_h,
        "kw": kcal_h * 1.163 / 1000,
        "btu_h": kcal_h * 3.96832,
    }


def _rotate(
    x: float,
    y: float,
    angle_degrees: float,
) -> Point:
    angle = math.radians(angle_degrees)
    if not angle:
        return x, y

    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )


def _transform_centered_point(
    local_x: float,
    local_y: float,
    obj: dict[str, Any],
) -> Point:
    """將以物件中心為原點的局部座標轉為畫布座標。"""
    scale_x = float(obj.get("scaleX", 1) or 1)
    scale_y = float(obj.get("scaleY", 1) or 1)
    left = float(obj.get("left", 0) or 0)
    top = float(obj.get("top", 0) or 0)
    angle = float(obj.get("angle", 0) or 0)

    x, y = _rotate(
        local_x * scale_x,
        local_y * scale_y,
        angle,
    )
    return x + left, y + top


def _origin_offset(
    obj: dict[str, Any],
    width: float,
    height: float,
) -> Point:
    origin_x = str(obj.get("originX", "left"))
    origin_y = str(obj.get("originY", "top"))

    x_map = {
        "left": width / 2,
        "center": 0.0,
        "right": -width / 2,
    }
    y_map = {
        "top": height / 2,
        "center": 0.0,
        "bottom": -height / 2,
    }
    return x_map.get(origin_x, width / 2), y_map.get(origin_y, height / 2)


def fabric_object_points(
    obj: dict[str, Any],
) -> list[Point]:
    """將 Fabric.js 物件還原為目前畫布上的實際座標。

    AI 候選框統一使用 Fabric Polygon，而不是 Path。
    Polygon 的 points 保留原始絕對座標，pathOffset 與 left/top
    都設定為多邊形中心，因此移動、縮放後可穩定還原。
    """
    object_type = str(obj.get("type", ""))

    if object_type == "polygon":
        path_offset = obj.get("pathOffset") or {"x": 0, "y": 0}
        offset_x = float(path_offset.get("x", 0) or 0)
        offset_y = float(path_offset.get("y", 0) or 0)

        return [
            _transform_centered_point(
                float(point["x"]) - offset_x,
                float(point["y"]) - offset_y,
                obj,
            )
            for point in obj.get("points", [])
        ]

    if object_type == "rect":
        width = float(obj.get("width", 0) or 0)
        height = float(obj.get("height", 0) or 0)
        offset_x, offset_y = _origin_offset(obj, width, height)

        local_corners = [
            (-width / 2 + offset_x, -height / 2 + offset_y),
            (width / 2 + offset_x, -height / 2 + offset_y),
            (width / 2 + offset_x, height / 2 + offset_y),
            (-width / 2 + offset_x, height / 2 + offset_y),
        ]
        return [
            _transform_centered_point(x, y, obj)
            for x, y in local_corners
        ]

    # 舊版 Path 僅作向下相容。正確扣除 pathOffset。
    if object_type == "path":
        path_offset = obj.get("pathOffset") or {"x": 0, "y": 0}
        offset_x = float(path_offset.get("x", 0) or 0)
        offset_y = float(path_offset.get("y", 0) or 0)

        points: list[Point] = []
        for command in obj.get("path", []):
            if (
                command
                and str(command[0]).upper() in {"M", "L"}
                and len(command) >= 3
            ):
                points.append(
                    _transform_centered_point(
                        float(command[1]) - offset_x,
                        float(command[2]) - offset_y,
                        obj,
                    )
                )

        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        return points

    return []


def fabric_line_endpoints(
    obj: dict[str, Any],
) -> tuple[Point, Point] | None:
    if obj.get("type") != "line":
        return None

    x1 = float(obj.get("x1", 0) or 0)
    y1 = float(obj.get("y1", 0) or 0)
    x2 = float(obj.get("x2", 0) or 0)
    y2 = float(obj.get("y2", 0) or 0)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return (
        _transform_centered_point(x1 - center_x, y1 - center_y, obj),
        _transform_centered_point(x2 - center_x, y2 - center_y, obj),
    )


def is_area_object(
    obj: dict[str, Any],
) -> bool:
    return (
        obj.get("type") in {"rect", "polygon", "path"}
        and len(fabric_object_points(obj)) >= 3
    )


def polygon_to_fabric_polygon(
    points: Iterable[Point],
    color: str = "#ff4b4b",
    stroke_width: int = 3,
    room_id: str | None = None,
    source: str = "auto",
) -> dict[str, Any]:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return {}

    polygon = Polygon(pts)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return {}
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)

    pts = [
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]

    min_x = min(x for x, _ in pts)
    max_x = max(x for x, _ in pts)
    min_y = min(y for _, y in pts)
    max_y = max(y for _, y in pts)

    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return {
        "type": "polygon",
        "version": "4.4.0",
        "originX": "center",
        "originY": "center",
        "left": center_x,
        "top": center_y,
        "width": width,
        "height": height,
        "fill": "rgba(0,0,0,0)",
        "stroke": color,
        "strokeWidth": int(stroke_width),
        "strokeUniform": True,
        "strokeLineCap": "round",
        "strokeLineJoin": "round",
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "visible": True,
        "selectable": True,
        "evented": True,
        "hasControls": True,
        "hasBorders": True,
        "objectCaching": False,
        "pathOffset": {
            "x": center_x,
            "y": center_y,
        },
        "points": [
            {"x": x, "y": y}
            for x, y in pts
        ],
        "room_id": room_id,
        "source": source,
    }


# 保留舊函式名稱，避免 app.py 或其他模組匯入失敗。
def polygon_to_fabric_path(
    points: Iterable[Point],
    color: str = "#ff4b4b",
    stroke_width: int = 3,
    room_id: str | None = None,
    source: str = "auto",
) -> dict[str, Any]:
    return polygon_to_fabric_polygon(
        points,
        color,
        stroke_width,
        room_id,
        source,
    )
