"""
當你目前的 floorplan_editor() 參數不是：
    floorplan_editor(image, rooms, width, height, key)

可以加這個轉接層，將新版 room polygons 轉成舊版資料格式。

請依你實際 editor 的欄位名稱微調。
"""


def adapt_rooms_for_editor(rooms):
    adapted = []
    for room in rooms:
        adapted.append(
            {
                "id": room["id"],
                "label": room.get("name", room["id"]),
                "points": [
                    {"x": float(x), "y": float(y)}
                    for x, y in room["points"]
                ],
                "stroke": "#e53935",
                "fill": "rgba(229,57,53,0.08)",
                "confidence": room.get("confidence", 0.0),
            }
        )
    return adapted
