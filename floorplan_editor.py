from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_FRONTEND_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.jsdelivr.net/npm/fabric@5.3.0/dist/fabric.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/streamlit-component-lib@2.0.0/dist/index.js"></script>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      font-family: sans-serif;
      background: #fff;
    }

    .toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      padding: 8px 0;
    }

    button,
    input[type="color"] {
      min-height: 36px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: white;
      padding: 6px 12px;
      cursor: pointer;
    }

    button.primary {
      background: #2563eb;
      color: white;
      border-color: #2563eb;
    }

    .status {
      color: #475569;
      font-size: 13px;
      margin-left: auto;
    }

    .canvas-wrap {
      border: 1px solid #cbd5e1;
      overflow: auto;
      background: #fff;
    }

    canvas {
      display: block;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <input id="color" type="color" value="#3B82F6" aria-label="框線顏色" />
    <button id="applyColor">套用顏色</button>
    <button id="delete">刪除選取</button>
    <button id="undo">復原</button>
    <button id="save" class="primary">套用修改</button>
    <span id="status" class="status">尚未修改</span>
  </div>

  <div class="canvas-wrap">
    <canvas id="canvas"></canvas>
  </div>

  <script>
    const Streamlit = window.Streamlit;
    const canvas = new fabric.Canvas("canvas", {
      selection: true,
      preserveObjectStacking: true
    });

    let history = [];
    let isRestoring = false;
    let initialized = false;

    function setFrameHeight() {
      Streamlit.setFrameHeight(document.body.scrollHeight + 8);
    }

    function saveHistory() {
      if (isRestoring) return;

      history.push(
        JSON.stringify(
          canvas.toJSON([
            "room_id",
            "room_name",
            "room_type",
            "confidence",
            "include_in_area",
            "source"
          ])
        )
      );

      if (history.length > 30) {
        history.shift();
      }

      document.getElementById("status").textContent =
        "有尚未套用的修改";
    }

    function createRoomObject(room) {
      const points = (room.points || []).map(point => ({
        x: Number(point[0]),
        y: Number(point[1])
      }));

      return new fabric.Polygon(points, {
        fill: "rgba(0,0,0,0)",
        stroke: room.color || "#ff6347",
        strokeWidth: 3,
        strokeUniform: true,
        objectCaching: false,
        transparentCorners: false,
        cornerStyle: "circle",
        cornerColor: "#ffffff",
        cornerStrokeColor: room.color || "#ff6347",
        room_id: room.room_id || "",
        room_name: room.room_name || "",
        room_type: room.room_type || "",
        confidence: room.confidence ?? null,
        include_in_area: room.include_in_area !== false,
        source: room.source || "openai"
      });
    }

    function extractPoints(object) {
      const matrix = object.calcTransformMatrix();

      return object.points.map(point => {
        const localPoint = new fabric.Point(
          point.x - object.pathOffset.x,
          point.y - object.pathOffset.y
        );

        const globalPoint =
          fabric.util.transformPoint(localPoint, matrix);

        return [globalPoint.x, globalPoint.y];
      });
    }

    function serializeRooms() {
      return canvas
        .getObjects()
        .filter(object => object.type === "polygon")
        .map(object => ({
          room_id: object.room_id || "",
          room_name: object.room_name || "",
          room_type: object.room_type || "",
          confidence: object.confidence ?? null,
          include_in_area: object.include_in_area !== false,
          source: object.source || "manual",
          color: object.stroke || "#ff6347",
          points: extractPoints(object)
        }));
    }

    function loadEditor(args) {
      const width = Number(args.width);
      const height = Number(args.height);

      canvas.setWidth(width);
      canvas.setHeight(height);
      canvas.clear();

      fabric.Image.fromURL(
        args.image_data_url,
        image => {
          image.set({
            left: 0,
            top: 0,
            selectable: false,
            evented: false,
            hoverCursor: "default"
          });

          canvas.add(image);
          canvas.sendToBack(image);

          (args.rooms || []).forEach(room => {
            canvas.add(createRoomObject(room));
          });

          canvas.renderAll();

          history = [
            JSON.stringify(
              canvas.toJSON([
                "room_id",
                "room_name",
                "room_type",
                "confidence",
                "include_in_area",
                "source"
              ])
            )
          ];

          initialized = true;
          document.getElementById("status").textContent =
            "可拖曳、拉伸、刪除";
          setFrameHeight();
        },
        { crossOrigin: "anonymous" }
      );
    }

    canvas.on("object:modified", saveHistory);

    document
      .getElementById("applyColor")
      .addEventListener("click", () => {
        const object = canvas.getActiveObject();

        if (!object || object.type !== "polygon") {
          return;
        }

        const color =
          document.getElementById("color").value;

        object.set({
          stroke: color,
          cornerStrokeColor: color
        });

        canvas.requestRenderAll();
        saveHistory();
      });

    document
      .getElementById("delete")
      .addEventListener("click", () => {
        const activeObjects = canvas.getActiveObjects();

        if (!activeObjects.length) {
          return;
        }

        activeObjects.forEach(object => {
          if (object.type === "polygon") {
            canvas.remove(object);
          }
        });

        canvas.discardActiveObject();
        canvas.requestRenderAll();
        saveHistory();
      });

    document
      .getElementById("undo")
      .addEventListener("click", () => {
        if (history.length <= 1) {
          return;
        }

        history.pop();
        const previous = history[history.length - 1];

        isRestoring = true;

        canvas.loadFromJSON(previous, () => {
          canvas.getObjects().forEach(object => {
            if (object.type === "image") {
              object.set({
                selectable: false,
                evented: false
              });
              canvas.sendToBack(object);
            }
          });

          canvas.renderAll();
          isRestoring = false;

          document.getElementById("status").textContent =
            "已復原，尚未套用";
        });
      });

    document
      .getElementById("save")
      .addEventListener("click", () => {
        Streamlit.setComponentValue({
          rooms: serializeRooms()
        });

        document.getElementById("status").textContent =
          "修改已套用";
      });

    Streamlit.events.addEventListener(
      Streamlit.RENDER_EVENT,
      event => {
        const args = event.detail.args;

        if (!initialized) {
          loadEditor(args);
        }
      }
    );

    Streamlit.setComponentReady();
    setFrameHeight();
  </script>
</body>
</html>
"""


def _prepare_component_directory() -> Path:
    """建立執行階段前端資料夾。

    不再依賴 GitHub 中一定要存在 frontend/index.html，
    可避免 Streamlit Cloud 因資料夾遺漏而啟動失敗。
    """
    component_dir = (
        Path(tempfile.gettempdir())
        / "floorplan_fabric_editor_component"
    )
    component_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_file = component_dir / "index.html"

    if (
        not index_file.exists()
        or index_file.read_text(encoding="utf-8")
        != _FRONTEND_HTML
    ):
        index_file.write_text(
            _FRONTEND_HTML,
            encoding="utf-8",
        )

    return component_dir


_COMPONENT_DIRECTORY = _prepare_component_directory()

_COMPONENT = components.declare_component(
    "floorplan_fabric_editor",
    path=str(_COMPONENT_DIRECTORY),
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
