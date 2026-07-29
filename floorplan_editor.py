from pathlib import Path
import streamlit.components.v1 as components

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

if not FRONTEND_DIR.exists():
    raise FileNotFoundError(f"找不到 frontend 資料夾：{FRONTEND_DIR}")

if not (FRONTEND_DIR / "index.html").exists():
    raise FileNotFoundError(
        f"找不到元件入口檔：{FRONTEND_DIR / 'index.html'}"
    )

_COMPONENT = components.declare_component(
    "floorplan_editor",
    path=str(FRONTEND_DIR),
)

def floorplan_editor(
    image_data_url: str,
    rooms: list[dict],
    zoom: float = 0.60,
    revision: int = 0,
    key: str | None = None,
):
    return _COMPONENT(
        image_data_url=image_data_url,
        rooms=rooms,
        zoom=zoom,
        revision=revision,
        key=key,
        default={"rooms": rooms, "zoom": zoom},
    )
