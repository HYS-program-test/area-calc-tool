from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

if not FRONTEND_DIR.is_dir():
    raise FileNotFoundError(
        "找不到自訂元件資料夾：{}".format(FRONTEND_DIR)
    )

if not INDEX_FILE.is_file():
    raise FileNotFoundError(
        "找不到自訂元件入口檔：{}".format(INDEX_FILE)
    )


_COMPONENT = components.declare_component(
    "floorplan_editor",
    path=str(FRONTEND_DIR),
)


def floorplan_editor(
    image_data_url: str,
    rooms: List[Dict[str, Any]],
    zoom: float = 0.60,
    revision: int = 0,
    key: Optional[str] = None,
) -> Any:
    """Render the floor-plan editor.

    `revision` is accepted explicitly because app.py uses it to distinguish
    a genuine external polygon reset from an ordinary Streamlit rerun.
    """
    return _COMPONENT(
        image_data_url=image_data_url,
        rooms=rooms,
        zoom=float(zoom),
        revision=int(revision),
        key=key,
        default={
            "rooms": rooms,
            "zoom": float(zoom),
        },
    )
