from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon


@dataclass
class DetectorConfig:
    """平面圖房間候選區域偵測參數。

    保留舊版 app.py 已使用的欄位名稱，因此可直接覆蓋原檔，不必同步修改 app.py。
    """

    adaptive_block_size: int = 31
    adaptive_c: int = 11

    # 牆線重建
    wall_line_length: int = 24
    wall_thickness: int = 3
    door_gap_px: int = 28

    # 候選空間篩選
    min_room_area_px: int = 7000
    max_room_area_ratio: float = 0.55
    polygon_epsilon_ratio: float = 0.006
    min_room_width_px: int = 25
    max_vertices: int = 60

    # 建築主體與雜訊排除
    roi_padding_px: int = 18
    min_building_ratio: float = 0.03
    max_building_ratio: float = 0.80
    annotation_density_threshold: float = 0.050
    border_reject_px: int = 4
    min_rectangularity: float = 0.16
    min_solidity: float = 0.55
    max_aspect_ratio: float = 10.0


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _remove_small_components(
    mask: np.ndarray,
    min_area: int = 30,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    output = np.zeros_like(mask)

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 255

    return output


def _safe_polygon(
    points: list[tuple[float, float]],
) -> Polygon | None:
    if len(points) < 3:
        return None

    poly = Polygon(points)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        return None

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda item: item.area)

    return poly


def _binary_ink(
    gray: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    """將圖面轉為深色線稿遮罩。"""

    block_size = _odd(config.adaptive_block_size)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(config.adaptive_c),
    )

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    ink = cv2.bitwise_and(
        adaptive,
        cv2.dilate(
            otsu,
            np.ones((2, 2), np.uint8),
        ),
    )

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    return _remove_small_components(
        ink,
        min_area=4,
    )


def _find_building_roi(
    ink: np.ndarray,
    config: DetectorConfig,
) -> tuple[int, int, int, int]:
    """找出最像建築主體的矩形範圍。

    尺寸線與基地界線通常是細長、低密度元件；
    建築主體通常具有較高線條密度、較多水平垂直交會，
    而且通常位於圖面主要區域。
    """

    height, width = ink.shape
    image_area = float(height * width)

    join = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9),
        ),
    )

    join = cv2.dilate(
        join,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (13, 13),
        ),
        iterations=1,
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            join,
            8,
        )
    )

    image_cx = width / 2.0
    image_cy = height / 2.0

    best_box: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for label in range(1, count):
        x, y, box_w, box_h, component_area = stats[label]

        box_area = float(box_w * box_h)
        ratio = box_area / image_area

        if ratio < config.min_building_ratio:
            continue

        if ratio > config.max_building_ratio:
            continue

        if min(box_w, box_h) < 80:
            continue

        original_ink_count = float(
            np.count_nonzero(
                ink[
                    y:y + box_h,
                    x:x + box_w,
                ]
            )
        )

        density = original_ink_count / max(
            box_area,
            1.0,
        )

        cx, cy = centroids[label]

        center_distance = np.hypot(
            (cx - image_cx) / max(width, 1),
            (cy - image_cy) / max(height, 1),
        )

        center_score = max(
            0.20,
            1.0 - center_distance,
        )

        aspect = max(
            box_w / max(box_h, 1),
            box_h / max(box_w, 1),
        )

        aspect_score = 1.0 / max(
            1.0,
            aspect / 3.0,
        )

        roi = ink[
            y:y + box_h,
            x:x + box_w,
        ]

        horizontal = cv2.morphologyEx(
            roi,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (18, 1),
            ),
        )

        vertical = cv2.morphologyEx(
            roi,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, 18),
            ),
        )

        junctions = cv2.bitwise_and(
            cv2.dilate(
                horizontal,
                np.ones((5, 5), np.uint8),
            ),
            cv2.dilate(
                vertical,
                np.ones((5, 5), np.uint8),
            ),
        )

        junction_score = min(
            2.0,
            0.5
            + np.count_nonzero(junctions)
            / 5000.0,
        )

        score = (
            component_area
            * max(density, 0.005)
            * center_score
            * aspect_score
            * junction_score
        )

        if score > best_score:
            best_score = score
            best_box = (
                int(x),
                int(y),
                int(box_w),
                int(box_h),
            )

    if best_box is None:
        return 0, 0, width, height

    x, y, box_w, box_h = best_box

    padding = max(
        0,
        int(config.roi_padding_px),
    )

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)

    x1 = min(
        width,
        x + box_w + padding,
    )

    y1 = min(
        height,
        y + box_h + padding,
    )

    return (
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )


def _roi_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = shape
    x, y, box_w, box_h = box

    mask = np.zeros(
        (height, width),
        np.uint8,
    )

    mask[
        y:y + box_h,
        x:x + box_w,
    ] = 255

    return mask


def _supported_orthogonal_lines(
    ink: np.ndarray,
    roi_mask: np.ndarray,
    config: DetectorConfig,
) -> dict[str, np.ndarray]:
    """擷取有局部密度或交會點支援的水平、垂直線。

    孤立尺寸線雖然很長，但附近通常沒有足夠牆角交會，
    因此不直接把所有長線都視為牆。
    """

    masked_ink = cv2.bitwise_and(
        ink,
        roi_mask,
    )

    line_length = max(
        10,
        int(config.wall_line_length),
    )

    horizontal = cv2.morphologyEx(
        masked_ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (line_length, 1),
        ),
    )

    vertical = cv2.morphologyEx(
        masked_ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, line_length),
        ),
    )

    horizontal_support = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (7, 3),
        ),
    )

    vertical_support = cv2.dilate(
        vertical,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 7),
        ),
    )

    junctions = cv2.bitwise_and(
        horizontal_support,
        vertical_support,
    )

    junction_support = cv2.dilate(
        junctions,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (25, 25),
        ),
    )

    local_density = cv2.boxFilter(
        (masked_ink > 0).astype(np.float32),
        ddepth=-1,
        ksize=(21, 21),
        normalize=True,
    )

    dense_support = np.zeros_like(
        masked_ink,
    )

    dense_support[
        local_density
        >= float(
            config.annotation_density_threshold
        )
    ] = 255

    dense_support = cv2.dilate(
        dense_support,
        np.ones((3, 3), np.uint8),
    )

    support = cv2.bitwise_or(
        dense_support,
        junction_support,
    )

    supported_horizontal = cv2.bitwise_and(
        horizontal,
        support,
    )

    supported_vertical = cv2.bitwise_and(
        vertical,
        support,
    )

    seed = cv2.bitwise_or(
        supported_horizontal,
        supported_vertical,
    )

    near_seed = cv2.dilate(
        seed,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (11, 11),
        ),
    )

    wall_ink = cv2.bitwise_and(
        masked_ink,
        near_seed,
    )

    wall_seed = cv2.bitwise_or(
        seed,
        wall_ink,
    )

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "junctions": junctions,
        "dense_support": dense_support,
        "wall_seed": wall_seed,
    }


def _build_wall_mask(
    ink: np.ndarray,
    roi_mask: np.ndarray,
    config: DetectorConfig,
) -> dict[str, np.ndarray]:
    parts = _supported_orthogonal_lines(
        ink,
        roi_mask,
        config,
    )

    walls = parts["wall_seed"]

    thickness = max(
        1,
        int(config.wall_thickness),
    )

    walls = cv2.dilate(
        walls,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (thickness, thickness),
        ),
        iterations=1,
    )

    ys, xs = np.where(
        roi_mask > 0,
    )

    if len(xs):
        roi_w = xs.max() - xs.min() + 1
        roi_h = ys.max() - ys.min() + 1

        adaptive_cap = max(
            8,
            int(
                round(
                    min(roi_w, roi_h)
                    * 0.035
                )
            ),
        )
    else:
        adaptive_cap = 16

    # 即使舊版 UI 傳入 28，也限制實際封閉距離不超過 18。
    effective_gap = max(
        5,
        min(
            int(config.door_gap_px),
            adaptive_cap,
            18,
        ),
    )

    close_h = cv2.morphologyEx(
        walls,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (effective_gap, 3),
        ),
    )

    close_v = cv2.morphologyEx(
        walls,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, effective_gap),
        ),
    )

    walls = cv2.bitwise_or(
        close_h,
        close_v,
    )

    walls = cv2.morphologyEx(
        walls,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
    )

    walls = _remove_small_components(
        walls,
        min_area=35,
    )

    walls = cv2.bitwise_and(
        walls,
        roi_mask,
    )

    parts["walls"] = walls
    parts["effective_gap"] = np.array(
        [[effective_gap]],
        dtype=np.uint8,
    )

    return parts


def _interior_regions(
    walls: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    free_space = cv2.bitwise_and(
        cv2.bitwise_not(walls),
        roi_mask,
    )

    height, width = free_space.shape

    flooded = free_space.copy()

    flood_mask = np.zeros(
        (height + 2, width + 2),
        np.uint8,
    )

    contours, _ = cv2.findContours(
        roi_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        x, y, box_w, box_h = cv2.boundingRect(
            max(
                contours,
                key=cv2.contourArea,
            )
        )

        step_x = max(
            1,
            box_w // 120,
        )

        step_y = max(
            1,
            box_h // 120,
        )

        seeds: list[tuple[int, int]] = []

        for px in range(
            x,
            x + box_w,
            step_x,
        ):
            seeds.extend(
                [
                    (px, y),
                    (px, y + box_h - 1),
                ]
            )

        for py in range(
            y,
            y + box_h,
            step_y,
        ):
            seeds.extend(
                [
                    (x, py),
                    (x + box_w - 1, py),
                ]
            )

        for px, py in seeds:
            if (
                0 <= px < width
                and 0 <= py < height
                and flooded[py, px] == 255
            ):
                cv2.floodFill(
                    flooded,
                    flood_mask,
                    (px, py),
                    128,
                )

    interiors = np.zeros_like(
        free_space,
    )

    interiors[
        flooded == 255
    ] = 255

    interiors = cv2.morphologyEx(
        interiors,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        ),
    )

    return interiors


def _touches_roi_border(
    stats_row: np.ndarray,
    roi_box: tuple[int, int, int, int],
    margin: int,
) -> bool:
    x, y, box_w, box_h = [
        int(value)
        for value in stats_row[:4]
    ]

    rx, ry, rw, rh = roi_box

    return (
        x <= rx + margin
        or y <= ry + margin
        or x + box_w >= rx + rw - margin
        or y + box_h >= ry + rh - margin
    )


def _component_to_polygon(
    component: np.ndarray,
    config: DetectorConfig,
) -> list[tuple[float, float]] | None:
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    contour_area = float(
        cv2.contourArea(contour)
    )

    if contour_area <= 0:
        return None

    x, y, box_w, box_h = cv2.boundingRect(
        contour
    )

    box_area = float(
        box_w * box_h
    )

    aspect = max(
        box_w / max(box_h, 1),
        box_h / max(box_w, 1),
    )

    rectangularity = (
        contour_area
        / max(box_area, 1.0)
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = float(
        cv2.contourArea(hull)
    )

    solidity = (
        contour_area
        / max(hull_area, 1.0)
    )

    if (
        min(box_w, box_h)
        < config.min_room_width_px
    ):
        return None

    if aspect > config.max_aspect_ratio:
        return None

    if (
        rectangularity
        < config.min_rectangularity
    ):
        return None

    if solidity < config.min_solidity:
        return None

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = max(
        1.0,
        float(
            config.polygon_epsilon_ratio
        )
        * perimeter,
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    points = [
        (
            float(item[0][0]),
            float(item[0][1]),
        )
        for item in approx
    ]

    if len(points) < 3:
        return None

    if len(points) > config.max_vertices:
        return None

    poly = _safe_polygon(
        points
    )

    if poly is None:
        return None

    return [
        (
            float(px),
            float(py),
        )
        for px, py
        in list(
            poly.exterior.coords
        )[:-1]
    ]


def detect_room_polygons(
    image: Image.Image,
    config: DetectorConfig | None = None,
) -> tuple[
    list[list[tuple[float, float]]],
    dict[str, Any],
]:
    """從平面圖建立房間候選多邊形。"""

    config = config or DetectorConfig()

    rgb = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    ink = _binary_ink(
        gray,
        config,
    )

    roi_box = _find_building_roi(
        ink,
        config,
    )

    building_mask = _roi_mask(
        ink.shape,
        roi_box,
    )

    masked_ink = cv2.bitwise_and(
        ink,
        building_mask,
    )

    wall_parts = _build_wall_mask(
        masked_ink,
        building_mask,
        config,
    )

    walls = wall_parts["walls"]

    interiors = _interior_regions(
        walls,
        building_mask,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            interiors,
            8,
        )
    )

    _, _, roi_w, roi_h = roi_box

    roi_area = float(
        max(
            roi_w * roi_h,
            1,
        )
    )

    polygons: list[
        list[tuple[float, float]]
    ] = []

    for label in range(1, count):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        box_w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )

        box_h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        if area < int(
            config.min_room_area_px
        ):
            continue

        if area > (
            roi_area
            * float(
                config.max_room_area_ratio
            )
        ):
            continue

        if min(
            box_w,
            box_h,
        ) < int(
            config.min_room_width_px
        ):
            continue

        if _touches_roi_border(
            stats[label],
            roi_box,
            int(
                config.border_reject_px
            ),
        ):
            continue

        component = np.zeros_like(
            interiors
        )

        component[
            labels == label
        ] = 255

        points = _component_to_polygon(
            component,
            config,
        )

        if not points:
            continue

        poly = _safe_polygon(
            points
        )

        if poly is None:
            continue

        if poly.area < float(
            config.min_room_area_px
        ):
            continue

        polygons.append(
            points
        )

    polygons.sort(
        key=lambda pts: (
            min(
                y
                for _, y in pts
            ),
            min(
                x
                for x, _ in pts
            ),
        )
    )

    debug: dict[str, Any] = {
        "gray": gray,
        "ink": ink,
        "building_mask": building_mask,
        "masked_ink": masked_ink,
        "horizontal": wall_parts[
            "horizontal"
        ],
        "vertical": wall_parts[
            "vertical"
        ],
        "junctions": wall_parts[
            "junctions"
        ],
        "dense_support": wall_parts[
            "dense_support"
        ],
        "walls": walls,
        "interiors": interiors,
        "roi_box": roi_box,
        "effective_gap": int(
            wall_parts[
                "effective_gap"
            ][0, 0]
        ),
    }

    return polygons, debug


def crop_to_main_floorplan(
    image: Image.Image,
    padding: int = 25,
) -> Image.Image:
    """裁切出建築主體。

    舊版只挑最大的連通元件，容易把基地線或尺寸線當成主體；
    新版改用線條密度、中心位置及交會點評分。
    """

    rgb = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    config = DetectorConfig(
        roi_padding_px=max(
            0,
            int(padding),
        )
    )

    ink = _binary_ink(
        gray,
        config,
    )

    x, y, box_w, box_h = (
        _find_building_roi(
            ink,
            config,
        )
    )

    original_area = (
        image.width
        * image.height
    )

    crop_area = (
        box_w
        * box_h
    )

    if crop_area < (
        original_area * 0.03
    ):
        return image

    if crop_area > (
        original_area * 0.95
    ):
        return image

    return image.crop(
        (
            x,
            y,
            x + box_w,
            y + box_h,
        )
    )
