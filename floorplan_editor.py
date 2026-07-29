from pathlib import Path
import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "floorplan_editor",
    path=str(Path(__file__).parent / "frontend"),
)

def floorplan_editor(image_data_url: str, rooms: list[dict], zoom: float = 0.60, key: str | None = None):
    return _COMPONENT(
        image_data_url=image_data_url,
        rooms=rooms,
        zoom=zoom,
        key=key,
        default={"rooms": rooms, "zoom": zoom},
    )
