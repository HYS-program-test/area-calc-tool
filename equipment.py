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


def load_equipment(
    excel_path: str | Path,
) -> dict[str, list[dict]]:
    """讀取固定放在 GitHub 根目錄的大金設備報價單。

    已確認欄位：
    A 類別、B 類型、C 型號、G 冷氣能力 kW、
    I/J/K 連結機型1/2/3、O 連接指數（VRV 算連結率用，取代舊版用型號數字用猜的）。

    A 欄類別支援：
    - VRV室外機 / VRV內機 / VRV室內機
    - 家用一對一室內機 / 家用一對一室外機
    - 家用一對多室內機 / 家用一對多室外機
    - 商用一對一室內機 / 商用一對一室外機

    回傳 dict，key 固定為：
    vrv_indoor, vrv_outdoor, home_indoor, home_outdoor,
    home_multi_indoor, home_multi_outdoor, commercial_indoor, commercial_outdoor
    """
    path = Path(excel_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到設備報價單：{path.name}。"
            "請將 Excel 放在 GitHub 專案根目錄。"
        )

    workbook = pd.ExcelFile(path)
    buckets: dict[str, list[dict]] = {
        "vrv_indoor": [],
        "vrv_outdoor": [],
        "home_indoor": [],
        "home_outdoor": [],
        "home_multi_indoor": [],
        "home_multi_outdoor": [],
        "commercial_indoor": [],
        "commercial_outdoor": [],
    }

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

            connection_index = _to_number(
                row.iloc[14] if len(row) > 14 else None
            )

            item = {
                "category": category,
                "type": equipment_type,
                "model": model,
                "capacity_kw": float(capacity_kw),
                "connection_models": connection_models,
                "connection_index": connection_index,
                "sheet": sheet_name,
                "excel_row": row_index + 1,
            }

            normalized = category.replace(" ", "")

            if "VRV室外機" in normalized:
                buckets["vrv_outdoor"].append(item)
            elif "VRV內機" in normalized or "VRV室內機" in normalized:
                buckets["vrv_indoor"].append(item)
            elif "家用一對多室外機" in normalized:
                buckets["home_multi_outdoor"].append(item)
            elif "家用一對多室內機" in normalized:
                buckets["home_multi_indoor"].append(item)
            elif "家用一對一室外機" in normalized:
                buckets["home_outdoor"].append(item)
            elif "家用一對一室內機" in normalized:
                buckets["home_indoor"].append(item)
            elif "商用一對一室外機" in normalized:
                buckets["commercial_outdoor"].append(item)
            elif "商用一對一室內機" in normalized:
                buckets["commercial_indoor"].append(item)

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

    return {key: deduplicate(items) for key, items in buckets.items()}


def find_closest_outdoor_1to1(
    indoor_model: str,
    outdoor_family: list[dict],
) -> dict:
    """家用一對一／商用一對一用：室外機＝同一個家族（家用配家用、商用配商用）裡，
    型號數字（從型號文字抽出來的，跟舊版 model_number() 邏輯一樣）最接近室內機的那一台。
    不是 VRV 那種連結率算法，純粹一對一配對。"""
    indoor_number = model_number(indoor_model)
    if indoor_number is None or not outdoor_family:
        return {"model": "", "capacity_kw": None}

    best_unit = None
    best_diff = None
    for unit in outdoor_family:
        unit_number = model_number(unit["model"])
        if unit_number is None:
            continue
        diff = abs(unit_number - indoor_number)
        if (
            best_diff is None
            or diff < best_diff
            or (diff == best_diff and unit["capacity_kw"] < best_unit["capacity_kw"])
        ):
            best_unit = unit
            best_diff = diff

    if best_unit is None:
        return {"model": "", "capacity_kw": None}
    return {"model": best_unit["model"], "capacity_kw": best_unit["capacity_kw"]}


def recommend_home_multi_outdoor(
    indoor_rows: list[dict],
    outdoor_units: list[dict],
    indoor_units: list[dict],
) -> dict:
    """家用一對多用：這是「一台室外機接多台室內機」，邏輯上跟 VRV 一樣要按分組處理，
    不是一對一。配對規則（根據原廠能力組合表）：室內機容量加總可以略大於室外機額定容量，
    取加總後最接近、且不超過太多的那一台室外機——實際做法是找「額定容量 ≤ 室內機加總」
    裡面最大的那一台；如果室內機加總比所有室外機都小，就退回容量最小的那一台頂著用。"""
    total_indoor_kw = 0.0
    for row in indoor_rows:
        unit = _find_unit_by_model(row.get("indoor_model", ""), indoor_units)
        quantity = row.get("indoor_quantity", 1) or 1
        if unit:
            total_indoor_kw += unit["capacity_kw"] * float(quantity)

    if total_indoor_kw <= 0 or not outdoor_units:
        return {"model": "", "capacity_kw": None, "total_indoor_kw": round(total_indoor_kw, 2)}

    eligible = [u for u in outdoor_units if u["capacity_kw"] <= total_indoor_kw]
    best_unit = max(eligible, key=lambda u: u["capacity_kw"]) if eligible else min(
        outdoor_units, key=lambda u: u["capacity_kw"]
    )

    return {
        "model": best_unit["model"],
        "capacity_kw": best_unit["capacity_kw"],
        "total_indoor_kw": round(total_indoor_kw, 2),
    }


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
    """備援用：Excel O 欄（連接指數）缺值時，退回用型號文字猜一個數字。
    正常情況下不會走到這裡——有 O 欄資料一律優先用 O 欄的實際數值。"""
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


# 小型 VRV 室外機：連結率算法跟一般 VRV 不一樣（不除以 25，直接用能力指數比），
# 型號比對時忽略大小寫與前後空白。
SMALL_VRV_OUTDOOR_MODELS = {
    "RXYCQ4AVET", "RXYCQ5AVET", "RXYCQ6AVET",
    "RXYMQ8TTLT", "RXYMQ10TTLT",
    "RXYMQ6TVET", "RXYMQ8TVET", "RXYMQ10TVET",
}


def _is_small_vrv(model: str) -> bool:
    return _clean_text(model).upper().replace(" ", "") in SMALL_VRV_OUTDOOR_MODELS


def _find_unit_by_model(model: str, units: list[dict]) -> dict | None:
    target = _clean_text(model).upper()
    for unit in units:
        if _clean_text(unit.get("model", "")).upper() == target:
            return unit
    return None


def _connection_index_for(model: str, units: list[dict]) -> float | None:
    """優先讀 Excel O 欄（連接指數）的實際數值；找不到這個型號或 O 欄是空的，
    才退回用型號文字猜一個數字（並不保證準確，只是避免整個計算掛掉）。"""
    unit = _find_unit_by_model(model, units)
    if unit is not None and unit.get("connection_index") is not None:
        return float(unit["connection_index"])
    return model_number(model)


def calculate_connection_rate(
    indoor_rows: list[dict],
    outdoor_model: str,
    indoor_units: list[dict],
    outdoor_units: list[dict],
) -> float | None:
    outdoor_index = _connection_index_for(outdoor_model, outdoor_units)
    if not outdoor_index or outdoor_index <= 0:
        return None

    total_indoor_index = 0.0
    for row in indoor_rows:
        indoor_index = _connection_index_for(
            row.get("indoor_model", ""), indoor_units
        )
        quantity = row.get("indoor_quantity", 1) or 1

        if indoor_index:
            total_indoor_index += indoor_index * float(quantity)

    if total_indoor_index <= 0:
        return None

    if _is_small_vrv(outdoor_model):
        # 小型 VRV：連結率% = 室內機能力指數加總 / 室外機能力指數（不除以25）
        return total_indoor_index / outdoor_index * 100.0

    # 一般 VRV：連結率% = 室內機能力指數加總 / 25 / 室外機數字
    return total_indoor_index / 25.0 / outdoor_index * 100.0


def recommend_outdoor(
    indoor_rows: list[dict],
    outdoor_units: list[dict],
    indoor_units: list[dict],
    min_rate: float = 105.0,
    max_rate: float = 110.0,
) -> dict:
    candidates: list[dict] = []
    target = (min_rate + max_rate) / 2

    for unit in outdoor_units:
        rate = calculate_connection_rate(
            indoor_rows,
            unit["model"],
            indoor_units,
            outdoor_units,
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
