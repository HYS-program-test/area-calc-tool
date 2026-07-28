from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _red_mask(image: Image.Image) -> np.ndarray:
    rgb = _to_rgb_array(image)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # 容許壓縮與生成造成的近紅色。
    mask = (
        (red >= 180)
        & (red >= green + 70)
        & (red >= blue + 70)
        & (green <= 120)
        & (blue <= 120)
    ).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return mask


def extract_red_polygons(
    annotated_image: Image.Image,
    min_area: float = 1500.0,
) -> list[dict[str, Any]]:
    """
    從 AI 回傳圖擷取紅色封閉線，轉成近似 Polygon。
    """
    mask = _red_mask(annotated_image)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour,
            epsilon=0.015 * perimeter,
            closed=True,
        )

        points = [
            [float(point[0][0]), float(point[0][1])]
            for point in approx
        ]

        if len(points) < 3:
            continue

        polygons.append(
            {
                "id": f"R{len(polygons) + 1}",
                "points": points,
                "area_pixels": area,
            }
        )

    polygons.sort(
        key=lambda item: (
            min(point[1] for point in item["points"]),
            min(point[0] for point in item["points"]),
        )
    )

    for index, polygon in enumerate(polygons):
        polygon["id"] = chr(65 + index) if index < 26 else f"R{index + 1}"

    return polygons


def draw_polygons(
    source_image: Image.Image,
    polygons: list[dict[str, Any]],
) -> Image.Image:
    overlay = source_image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for polygon in polygons:
        points = [
            (float(x), float(y))
            for x, y in polygon["points"]
        ]

        if len(points) < 3:
            continue

        draw.line(
            points + [points[0]],
            fill=(255, 0, 0),
            width=5,
            joint="curve",
        )

        x, y = points[0]
        draw.text(
            (x + 6, y + 6),
            polygon["id"],
            fill=(255, 0, 0),
        )

    return overlay
