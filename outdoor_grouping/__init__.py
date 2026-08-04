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
    "outdoor_grouping",
    path=str(FRONTEND_DIR),
)


def outdoor_grouping(
    rows: List[Dict[str, Any]],
    revision: int = 0,
    key: Optional[str] = None,
) -> Any:
    """
    室外機分組編輯器：真正的合併儲存格，拖曳虛線可以自由調整哪幾個房間屬於同一台室外機，
    連結率／室外機型號會依分組即時重算（重算邏輯在 Python 端，這個元件只負責顯示分組跟
    回報使用者怎麼調整分組）。

    rows: [
        {
            "index": int,               # 房間在目前表格裡的順序（0-based）
            "label": str,                # 房間名稱，純顯示用
            "is_split": bool,             # 這個房間是不是「新分組」的起點
            "outdoor_model": str,         # 這個分組目前推薦的室外機型號（合併儲存格顯示用）
            "connection_rate": float | None,  # 這個分組目前的連結率（合併儲存格顯示用）
        }, ...
    ]

    revision 只是給呼叫端（app.py）自己記帳用，元件本身不會讀。

    回傳： {"splits": [{"index": int}, ...]}
    （目前每一組的起始房間 index，由小到大排序）
    """
    default_splits = [{"index": r["index"]} for r in rows if r.get("is_split")]
    return _COMPONENT(
        rows=rows,
        revision=int(revision),
        key=key,
        default={"splits": default_splits},
    )
