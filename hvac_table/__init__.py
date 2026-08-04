from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

if not FRONTEND_DIR.is_dir():
    raise FileNotFoundError("找不到自訂元件資料夾：{}".format(FRONTEND_DIR))
if not INDEX_FILE.is_file():
    raise FileNotFoundError("找不到自訂元件入口檔：{}".format(INDEX_FILE))

_COMPONENT = components.declare_component(
    "hvac_table",
    path=str(FRONTEND_DIR),
)


def hvac_table(
    rows: List[Dict[str, Any]],
    palette: List[str],
    revision: int = 0,
    key: Optional[str] = None,
) -> Any:
    """
    空調負荷計算結果表格，白底、顏色欄是真正的色塊下拉選單（不是色碼文字），
    室外機型號／連結率是真正的合併儲存格（rowspan）。

    rows: [
        {
            "room_id": int,
            "name": str,
            "color": str,               # hex，一定要在 palette 裡
            "area_m2": float, "area_ping": float,
            "per_ping_load": float,
            "total_heat_kw": float,
            "indoor_model": str,
            "indoor_quantity": int,
            "indoor_capacity_kw": float | None,
            "average_load": float | None,
            "outdoor_model": str,
            "connection_rate": float | None,
            "merge_group": int,          # 連續同一個值的列會合併成一格（室外機/連結率那欄）
        }, ...
    ]
    palette: 顏色下拉選單的選項（hex 字串列表）

    revision 只是給呼叫端（app.py）自己記帳用，元件本身不會讀。

    回傳： {
        "updates": [{"room_id":int,"name":str,"color":str,"per_ping_load":float,
                      "indoor_model":str,"indoor_quantity":int,"indoor_capacity_kw":float|None}, ...],
        "nonce": int,
    }
    """
    return _COMPONENT(
        rows=rows,
        palette=palette,
        revision=int(revision),
        key=key,
        default={"updates": [], "nonce": 0},
    )
