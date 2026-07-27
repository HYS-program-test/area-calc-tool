from __future__ import annotations
from pathlib import Path
from typing import Any
import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "floorplan_fabric_editor",
    path=str(Path(__file__).parent / "frontend"),
)

def floorplan_editor(*, image_data_url: str, width: int, height: int,
                     rooms: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return _COMPONENT(
        image_data_url=image_data_url,
        width=width,
        height=height,
        rooms=rooms,
        key=key,
        default=None,
    )
