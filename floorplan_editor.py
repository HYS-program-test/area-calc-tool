from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_HTML = r"""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; background:#fff; font-family:Arial,sans-serif; }
  #toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:8px 0; }
  button, input { min-height:36px; border:1px solid #cbd5e1; border-radius:8px; background:#fff; padding:6px 12px; }
  button { cursor:pointer; }
  button.primary { background:#2563eb; color:#fff; border-color:#2563eb; }
  #status { margin-left:auto; color:#475569; font-size:13px; }
  #wrap { border:1px solid #cbd5e1; overflow:auto; background:#fff; }
  svg { display:block; touch-action:none; user-select:none; }
  .room { fill:transparent; stroke-width:3; vector-effect:non-scaling-stroke; cursor:move; }
  .room.selected { stroke-width:5; }
  .handle { fill:#fff; stroke:#111827; stroke-width:1.5; cursor:crosshair; }
</style>
</head>
<body>
<div id="toolbar">
  <input id="color" type="color" value="#3B82F6">
  <button id="applyColor">套用顏色</button>
  <button id="delete">刪除選取</button>
  <button id="undo">復原</button>
  <button id="save" class="primary">套用修改</button>
  <span id="status">載入中</span>
</div>
<div id="wrap"><svg id="editor"></svg></div>

<script>
(() => {
  const svg = document.getElementById("editor");
  const NS = "http://www.w3.org/2000/svg";
  let args = null;
  let rooms = [];
  let selectedIndex = -1;
  let dragMode = null;
  let dragStart = null;
  let originalPoints = null;
  let activeVertex = -1;
  let history = [];

  function send(type, data={}) {
    window.parent.postMessage({isStreamlitMessage:true, type, ...data}, "*");
  }

  function ready() {
    send("streamlit:componentReady", {apiVersion:1});
  }

  function setHeight() {
    send("streamlit:setFrameHeight", {height:document.body.scrollHeight + 8});
  }

  function setValue(value) {
    send("streamlit:setComponentValue", {value});
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function pushHistory() {
    history.push(clone(rooms));
    if (history.length > 30) history.shift();
    document.getElementById("status").textContent = "有尚未套用的修改";
  }

  function pointString(points) {
    return points.map(p => `${p[0]},${p[1]}`).join(" ");
  }

  function clearSelection() {
    selectedIndex = -1;
    render();
  }

  function selectRoom(index) {
    selectedIndex = index;
    render();
  }

  function svgPoint(event) {
    const rect = svg.getBoundingClientRect();
    const scaleX = Number(args.width) / rect.width;
    const scaleY = Number(args.height) / rect.height;
    return [
      (event.clientX - rect.left) * scaleX,
      (event.clientY - rect.top) * scaleY
    ];
  }

  function render() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    svg.setAttribute("width", args.width);
    svg.setAttribute("height", args.height);
    svg.setAttribute("viewBox", `0 0 ${args.width} ${args.height}`);

    const image = document.createElementNS(NS, "image");
    image.setAttribute("href", args.image_data_url);
    image.setAttribute("x", "0");
    image.setAttribute("y", "0");
    image.setAttribute("width", args.width);
    image.setAttribute("height", args.height);
    image.setAttribute("preserveAspectRatio", "none");
    image.style.pointerEvents = "none";
    svg.appendChild(image);

    rooms.forEach((room, index) => {
      const polygon = document.createElementNS(NS, "polygon");
      polygon.setAttribute("points", pointString(room.points || []));
      polygon.setAttribute("stroke", room.color || "#ff6347");
      polygon.setAttribute("class", index === selectedIndex ? "room selected" : "room");
      polygon.dataset.index = String(index);

      polygon.addEventListener("pointerdown", event => {
        event.stopPropagation();
        selectedIndex = index;
        dragMode = "move";
        dragStart = svgPoint(event);
        originalPoints = clone(rooms[index].points);
        svg.setPointerCapture(event.pointerId);
        render();
      });

      svg.appendChild(polygon);

      if (index === selectedIndex) {
        (room.points || []).forEach((point, vertexIndex) => {
          const handle = document.createElementNS(NS, "circle");
          handle.setAttribute("cx", point[0]);
          handle.setAttribute("cy", point[1]);
          handle.setAttribute("r", "7");
          handle.setAttribute("class", "handle");

          handle.addEventListener("pointerdown", event => {
            event.stopPropagation();
            dragMode = "vertex";
            activeVertex = vertexIndex;
            dragStart = svgPoint(event);
            originalPoints = clone(rooms[index].points);
            svg.setPointerCapture(event.pointerId);
          });

          svg.appendChild(handle);
        });
      }
    });

    setHeight();
  }

  svg.addEventListener("pointerdown", event => {
    if (event.target === svg || event.target.tagName === "image") {
      clearSelection();
    }
  });

  svg.addEventListener("pointermove", event => {
    if (selectedIndex < 0 || !dragMode || !dragStart || !originalPoints) return;

    const current = svgPoint(event);
    const dx = current[0] - dragStart[0];
    const dy = current[1] - dragStart[1];

    if (dragMode === "move") {
      rooms[selectedIndex].points = originalPoints.map(p => [
        Math.max(0, Math.min(Number(args.width), p[0] + dx)),
        Math.max(0, Math.min(Number(args.height), p[1] + dy))
      ]);
    } else if (dragMode === "vertex" && activeVertex >= 0) {
      rooms[selectedIndex].points = clone(originalPoints);
      rooms[selectedIndex].points[activeVertex] = [
        Math.max(0, Math.min(Number(args.width), current[0])),
        Math.max(0, Math.min(Number(args.height), current[1]))
      ];
    }

    render();
  });

  svg.addEventListener("pointerup", () => {
    if (dragMode) pushHistory();
    dragMode = null;
    dragStart = null;
    originalPoints = null;
    activeVertex = -1;
  });

  document.getElementById("applyColor").addEventListener("click", () => {
    if (selectedIndex < 0) return;
    rooms[selectedIndex].color = document.getElementById("color").value;
    pushHistory();
    render();
  });

  document.getElementById("delete").addEventListener("click", () => {
    if (selectedIndex < 0) return;
    rooms.splice(selectedIndex, 1);
    selectedIndex = -1;
    pushHistory();
    render();
  });

  document.getElementById("undo").addEventListener("click", () => {
    if (history.length <= 1) return;
    history.pop();
    rooms = clone(history[history.length - 1]);
    selectedIndex = -1;
    document.getElementById("status").textContent = "已復原，尚未套用";
    render();
  });

  document.getElementById("save").addEventListener("click", () => {
    setValue({rooms: clone(rooms)});
    document.getElementById("status").textContent = "修改已套用";
  });

  window.addEventListener("message", event => {
    const data = event.data;
    if (!data || data.type !== "streamlit:render") return;

    args = data.args;
    rooms = clone(args.rooms || []);
    selectedIndex = -1;
    history = [clone(rooms)];
    document.getElementById("status").textContent = "可拖曳框線；拖曳頂點可調整形狀";
    render();
  });

  ready();
  setHeight();
})();
</script>
</body>
</html>
"""


def _component_dir() -> Path:
    directory = Path(tempfile.gettempdir()) / "floorplan_svg_editor_component"
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "index.html"
    index.write_text(_HTML, encoding="utf-8")
    return directory


_COMPONENT = components.declare_component(
    "floorplan_svg_editor",
    path=str(_component_dir()),
)


def floorplan_editor(
    *,
    image_data_url: str,
    width: int,
    height: int,
    rooms: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    return _COMPONENT(
        image_data_url=image_data_url,
        width=width,
        height=height,
        rooms=rooms,
        key=key,
        default=None,
    )
