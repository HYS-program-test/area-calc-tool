from __future__ import annotations

import io
import json
import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from openai_floorplan_analyzer import analyze_floorplan


st.set_page_config(
    page_title="OpenAI 平面圖空間辨識",
    layout="wide",
)

st.title("OpenAI 平面圖空間辨識")
st.caption(
    "上傳圖面後，整張圖會直接交給 OpenAI Vision 模型辨識空間 Polygon。"
)


def render_pdf_first_page(
    pdf_bytes: bytes,
    dpi: int = 220,
) -> Image.Image:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(document) == 0:
        document.close()
        raise ValueError("PDF 沒有頁面。")

    page = document[0]
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(pixmap.tobytes("png"))
    ).convert("RGB")

    document.close()
    return image


def read_uploaded_file(uploaded_file) -> Image.Image:
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return render_pdf_first_page(content, dpi=220)

    return Image.open(io.BytesIO(content)).convert("RGB")


def resize_for_api(
    image: Image.Image,
    max_side: int = 2200,
) -> tuple[Image.Image, float]:
    """
    避免超大 PDF Render 圖片造成傳輸及模型處理負擔。
    Polygon 座標以縮放後圖片為準，頁面底圖亦使用同一張圖片，
    因此不需要額外座標回推。
    """
    width, height = image.size
    longest = max(width, height)

    if longest <= max_side:
        return image, 1.0

    scale = max_side / longest
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    resized = image.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def draw_overlay(
    image: Image.Image,
    rooms: list[dict],
) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for room in rooms:
        points = [
            (float(x), float(y))
            for x, y in room["polygon"]
        ]

        if len(points) < 3:
            continue

        draw.line(
            points + [points[0]],
            fill="red",
            width=5,
            joint="curve",
        )

        label_x, label_y = points[0]
        area_text = (
            f'{room["area_m2"]:.2f} m²'
            if room.get("area_m2") is not None
            else "尚無比例"
        )

        draw.text(
            (label_x + 8, label_y + 8),
            f'{room["id"]} {room["name"]} | {area_text}',
            fill="red",
        )

    return overlay


uploaded_file = st.file_uploader(
    "上傳 PDF、PNG、JPG",
    type=["pdf", "png", "jpg", "jpeg"],
)

model = st.text_input(
    "OpenAI Vision 模型",
    value=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"),
)

if uploaded_file is not None:
    try:
        original_image = read_uploaded_file(uploaded_file)
        api_image, resize_scale = resize_for_api(
            original_image,
            max_side=2200,
        )

        st.write(
            f"送入 API 的圖片尺寸："
            f"{api_image.width} × {api_image.height}"
        )

        st.image(
            api_image,
            caption="將送入 OpenAI API 的完整圖面",
            use_container_width=True,
        )

        if st.button(
            "交給 OpenAI 自動辨識及框面積",
            type="primary",
            use_container_width=True,
        ):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("尚未設定 OPENAI_API_KEY。")
                st.stop()

            with st.spinner(
                "OpenAI 正在閱讀圖面、辨識室內空間並產生 Polygon……"
            ):
                result = analyze_floorplan(
                    api_image,
                    model=model,
                )

            st.session_state["openai_floorplan_result"] = result
            st.session_state["openai_floorplan_image"] = api_image

        result = st.session_state.get(
            "openai_floorplan_result"
        )
        result_image = st.session_state.get(
            "openai_floorplan_image"
        )

        if result and result_image:
            overlay = draw_overlay(
                result_image,
                result["rooms"],
            )

            st.subheader("OpenAI 辨識結果")
            st.image(
                overlay,
                caption="OpenAI Polygon 疊圖",
                use_container_width=True,
            )

            st.subheader("空間面積")

            if not result["rooms"]:
                st.warning("OpenAI 未回傳有效空間 Polygon。")
            else:
                rows = []
                for room in result["rooms"]:
                    rows.append(
                        {
                            "編號": room["id"],
                            "空間": room["name"],
                            "信心": round(
                                room["confidence"],
                                3,
                            ),
                            "像素面積": round(
                                room["area_pixels"],
                                2,
                            ),
                            "面積(m²)": (
                                round(room["area_m2"], 2)
                                if room["area_m2"] is not None
                                else None
                            ),
                        }
                    )

                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )

            if not result["scale"]["scale_found"]:
                st.warning(
                    "OpenAI 未能可靠辨識尺寸比例，因此目前只能顯示像素面積。"
                )
            else:
                st.success(
                    "比例尺辨識成功："
                    f'{result["scale"]["pixels_per_meter"]:.3f} pixels/m'
                )

            with st.expander("查看 OpenAI 回傳 JSON"):
                st.json(result)

            result_json = json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )

            st.download_button(
                "下載辨識結果 JSON",
                data=result_json,
                file_name="floorplan_openai_result.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as exc:
        st.exception(exc)
