from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EQUIPMENT_FILENAME = "大金空調價格表_設備報價單.xlsx"


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _to_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _find_header_row(df: pd.DataFrame) -> int | None:
    max_scan = min(len(df), 30)

    for index in range(max_scan):
        values = [
            _clean_text(value).replace("\n", "")
            for value in df.iloc[index].tolist()
        ]
        joined = "|".join(values)

        has_category = any("類別" in value for value in values[:3])
        has_model = any("型號" in value for value in values[:5])
        has_capacity = "冷氣能力" in joined

        if has_category and has_model and has_capacity:
            return index

    return None


def load_vrv_equipment(
    excel_path: str | Path,
) -> tuple[list[dict], list[dict]]:
    """讀取固定放在 GitHub 根目錄的大金設備報價單。

    已確認欄位：
    A 類別、B 類型、C 型號、G 冷氣能力 kW、
    I/J/K 連結機型1/2/3。
    """
    path = Path(excel_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到設備報價單：{path.name}。"
            "請將 Excel 放在 GitHub 專案根目錄。"
        )

    workbook = pd.ExcelFile(path)
    indoor_units: list[dict] = []
    outdoor_units: list[dict] = []

    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
            dtype=object,
        )

        header_row = _find_header_row(raw)
        start_row = header_row + 1 if header_row is not None else 0

        for row_index in range(start_row, len(raw)):
            row = raw.iloc[row_index]

            category = _clean_text(
                row.iloc[0] if len(row) > 0 else ""
            )
            equipment_type = _clean_text(
                row.iloc[1] if len(row) > 1 else ""
            )
            model = _clean_text(
                row.iloc[2] if len(row) > 2 else ""
            )
            capacity_kw = _to_number(
                row.iloc[6] if len(row) > 6 else None
            )

            if not model or capacity_kw is None or capacity_kw <= 0:
                continue

            connection_models: list[str] = []
            for column_index in (8, 9, 10):
                if len(row) <= column_index:
                    continue
                value = _clean_text(row.iloc[column_index])
                if value:
                    connection_models.append(value)

            item = {
                "category": category,
                "type": equipment_type,
                "model": model,
                "capacity_kw": float(capacity_kw),
                "connection_models": connection_models,
                "sheet": sheet_name,
                "excel_row": row_index + 1,
            }

            normalized = category.replace(" ", "")

            if "VRV室外機" in normalized:
                outdoor_units.append(item)
            elif (
                "VRV內機" in normalized
                or "VRV室內機" in normalized
            ):
                indoor_units.append(item)

    def deduplicate(items: list[dict]) -> list[dict]:
        by_model: dict[str, dict] = {}

        for item in items:
            existing = by_model.get(item["model"])
            if existing is None:
                by_model[item["model"]] = item
            elif (
                not existing.get("connection_models")
                and item.get("connection_models")
            ):
                by_model[item["model"]] = item

        return sorted(
            by_model.values(),
            key=lambda item: (
                item["capacity_kw"],
                item["model"],
            ),
        )

    return deduplicate(indoor_units), deduplicate(outdoor_units)


def recommend_indoor(
    demand_kw: float,
    indoor_units: list[dict],
) -> dict:
    if demand_kw <= 0 or not indoor_units:
        return {
            "model": "",
            "capacity_kw": None,
            "quantity": 1,
        }

    for unit in indoor_units:
        if unit["capacity_kw"] >= demand_kw:
            return {
                "model": unit["model"],
                "capacity_kw": unit["capacity_kw"],
                "quantity": 1,
                "type": unit.get("type", ""),
            }

    largest = indoor_units[-1]
    quantity = max(
        1,
        math.ceil(demand_kw / largest["capacity_kw"]),
    )
    return {
        "model": largest["model"],
        "capacity_kw": largest["capacity_kw"],
        "quantity": quantity,
        "type": largest.get("type", ""),
    }


def model_number(model: str) -> float | None:
    text = _clean_text(model).upper()

    preferred = re.search(
        r"(?:FX[A-Z]*|RXYQ|RZQ|RQQ|RQYQ|RXQ)"
        r"(\d+(?:\.\d+)?)",
        text,
    )
    if preferred:
        return float(preferred.group(1))

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def calculate_connection_rate(
    indoor_rows: list[dict],
    outdoor_model: str,
) -> float | None:
    outdoor_number = model_number(outdoor_model)
    if not outdoor_number or outdoor_number <= 0:
        return None

    total_indoor_index = 0.0
    for row in indoor_rows:
        indoor_number = model_number(
            row.get("indoor_model", "")
        )
        quantity = row.get("indoor_quantity", 1) or 1

        if indoor_number:
            total_indoor_index += (
                indoor_number * float(quantity)
            )

    if total_indoor_index <= 0:
        return None

    return (
        total_indoor_index
        / 25.0
        / outdoor_number
        * 100.0
    )


def recommend_outdoor(
    indoor_rows: list[dict],
    outdoor_units: list[dict],
    min_rate: float = 105.0,
    max_rate: float = 110.0,
) -> dict:
    candidates: list[dict] = []
    target = (min_rate + max_rate) / 2

    for unit in outdoor_units:
        rate = calculate_connection_rate(
            indoor_rows,
            unit["model"],
        )
        if rate is None:
            continue

        in_range = min_rate <= rate <= max_rate

        if in_range:
            distance = abs(rate - target)
        elif rate < min_rate:
            distance = min_rate - rate
        else:
            distance = rate - max_rate

        candidates.append({
            **unit,
            "connection_rate": rate,
            "in_target_range": in_range,
            "distance": distance,
        })

    if not candidates:
        return {
            "model": "",
            "capacity_kw": None,
            "connection_rate": None,
            "in_target_range": False,
            "connection_models": [],
        }

    candidates.sort(
        key=lambda item: (
            not item["in_target_range"],
            item["distance"],
            item["capacity_kw"],
        )
    )
    return candidates[0]
