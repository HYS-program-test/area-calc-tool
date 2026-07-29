from pathlib import Path
from pdf_utils import render_pdf_page
from hvac import calculate_rows

pdf = Path("sample_3F.pdf").read_bytes()
img, meta = render_pdf_page(pdf, 0, 220)
rooms = [
    {"id":1,"name":"區域1","room_type":"一般辦公室","color":"#ef4444","unit_load":120,"included":True,
     "points":[{"x":100,"y":100},{"x":300,"y":100},{"x":300,"y":300},{"x":100,"y":300}]},
    {"id":2,"name":"區域2","room_type":"會議室","color":"#f59e0b","unit_load":150,"included":True,
     "points":[{"x":500,"y":100},{"x":800,"y":100},{"x":800,"y":400},{"x":500,"y":400}]},
]
rows = calculate_rows(rooms, img.width, img.height, 10000)
print({"meta":meta,"rows":rows})
