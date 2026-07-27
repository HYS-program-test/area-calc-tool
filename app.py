from __future__ import annotations
import base64, io, json, re
from pathlib import Path
import fitz, pandas as pd, streamlit as st
from PIL import Image, ImageDraw
from floorplan_detector import crop_to_main_floorplan
from floorplan_editor import floorplan_editor
from geometry_utils import cooling_load, pixel_area_to_m2, polygon_area_px2
from openai_room_detector import AIRoomDetectionOptions, detect_rooms_with_openai

st.set_page_config(page_title="AI 平面圖空調設備選型",page_icon="❄️",layout="wide")
LOAD_OPTIONS=list(range(400,1300,100));DEFAULT_MODEL="gpt-4.1";DEFAULT_DPI=200;DEFAULT_WIDTH=1100

def init_state():
    for k,v in {"file_key":None,"rooms":[],"ai_result":None,"editor_version":0,"px_per_meter":None}.items():
        if k not in st.session_state:st.session_state[k]=v

@st.cache_data(show_spinner=False)
def load_pdf_page(data,page_index,dpi):
    doc=fitz.open(stream=data,filetype="pdf");page=doc.load_page(page_index);text=page.get_text();scale=None
    for pattern in [r"1\s*[:：]\s*(\d+)",r"1\s*/\s*(\d+)"]:
        m=re.search(pattern,text)
        if m and 10<=int(m.group(1))<=5000:scale=int(m.group(1));break
    pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False)
    image=Image.frombytes("RGB",(pix.width,pix.height),pix.samples);doc.close();return image,scale

@st.cache_data(show_spinner=False)
def load_image(data):return Image.open(io.BytesIO(data)).convert("RGB")

def resize_image(image,max_width):
    s=min(1.0,max_width/image.width)
    return image.copy() if s>=1 else image.resize((round(image.width*s),round(image.height*s)),Image.Resampling.LANCZOS)

def image_data_url(image):
    b=io.BytesIO();image.convert("RGB").save(b,format="JPEG",quality=90)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

def export_pdf(image,rooms,px_per_meter):
    out=image.convert("RGB").copy();draw=ImageDraw.Draw(out)
    for room in rooms:
        pts=[(round(x),round(y)) for x,y in room["points"]]
        if len(pts)<3:continue
        color=room.get("color","#ff6347");draw.line(pts+[pts[0]],fill=color,width=5)
        area=pixel_area_to_m2(polygon_area_px2(room["points"]),px_per_meter)
        name=room.get("room_name") or room.get("room_id","");label=name if area is None else f"{name} {area:.2f} m²"
        cx=round(sum(x for x,_ in pts)/len(pts));cy=round(sum(y for _,y in pts)/len(pts))
        draw.rectangle((cx-4,cy-4,cx+max(70,len(label)*8),cy+18),fill="white",outline=color);draw.text((cx,cy),label,fill=color)
    b=io.BytesIO();out.save(b,format="PDF",resolution=200);return b.getvalue()

init_state()
st.markdown("## ❄️ AI 平面圖空調設備選型")
st.caption("每張圖先由 GPT 完整辨識，再逐房間放大判讀。拖曳、拉伸、改色與刪除都在瀏覽器內完成，按『套用修改』才回傳 Streamlit。")
uploaded=st.file_uploader("上傳平面圖 PDF／PNG／JPG",type=["pdf","png","jpg","jpeg"])
if uploaded is None:st.info("請先上傳平面圖。");st.stop()
data=uploaded.getvalue();is_pdf=uploaded.name.lower().endswith(".pdf");page_index=0
if is_pdf:
    doc=fitz.open(stream=data,filetype="pdf");count=doc.page_count;doc.close()
    if count>1:page_index=st.selectbox("PDF 頁面",range(count),format_func=lambda i:f"第 {i+1} 頁")
file_key=f"{uploaded.name}:{len(data)}:{hash(data)}:{page_index}"
if st.session_state.file_key!=file_key:
    st.session_state.file_key=file_key;st.session_state.rooms=[];st.session_state.ai_result=None;st.session_state.editor_version+=1;st.session_state.px_per_meter=None
if is_pdf:source_image,auto_scale=load_pdf_page(data,page_index,DEFAULT_DPI)
else:source_image,auto_scale=load_image(data),None
source_image=crop_to_main_floorplan(source_image);display_image=resize_image(source_image,DEFAULT_WIDTH)
api_key=st.secrets.get("OPENAI_API_KEY","")
c1,c2=st.columns(2)
with c1:
    if st.button("✨ GPT 辨識並建立候選框",type="primary",use_container_width=True,disabled=not api_key):
        try:
            with st.spinner("GPT 正在完整判讀圖面並逐房間精修…"):
                result=detect_rooms_with_openai(api_key=api_key,image=display_image,model=DEFAULT_MODEL,options=AIRoomDetectionOptions())
            colors=["#FF6347","#3B82F6","#22C55E","#F59E0B","#A855F7","#06B6D4","#EC4899","#84CC16"]
            st.session_state.rooms=[{**r,"color":colors[i%8]} for i,r in enumerate(result["rooms"])]
            st.session_state.ai_result=result;st.session_state.editor_version+=1;st.rerun()
        except Exception as e:st.error(f"AI 辨識失敗：{e}")
with c2:
    if st.button("清空全部框線",use_container_width=True):
        st.session_state.rooms=[];st.session_state.ai_result=None;st.session_state.editor_version+=1;st.rerun()
if not api_key:st.error("尚未設定 OPENAI_API_KEY。")
if st.session_state.ai_result:
    r=st.session_state.ai_result;m1,m2,m3=st.columns(3);m1.metric("接受候選空間",len(r.get("rooms",[])));m2.metric("排除候選空間",len(r.get("rejected_rooms",[])));m3.metric("圖面品質",r.get("image_assessment",{}).get("quality","未知"))
st.markdown("### 單一可編輯平面圖")
editor_result=floorplan_editor(image_data_url=image_data_url(display_image),width=display_image.width,height=display_image.height,rooms=st.session_state.rooms,key=f"fabric_editor_{st.session_state.editor_version}")
if editor_result and "rooms" in editor_result:
    st.session_state.rooms=editor_result["rooms"];st.success("框線修改已套用。")
st.markdown("### 空間資料")
if st.session_state.rooms:
    df=pd.DataFrame([{"編號":r.get("room_id",""),"空間名稱":r.get("room_name",""),"空間類型":r.get("room_type",""),"納入面積":r.get("include_in_area",True),"信心分數":r.get("confidence")} for r in st.session_state.rooms])
    edited=st.data_editor(df,hide_index=True,use_container_width=True,disabled=["編號","信心分數"],key="room_table")
    lookup={row["編號"]:row for row in edited.to_dict("records")}
    for room in st.session_state.rooms:
        row=lookup.get(room.get("room_id"))
        if row:room["room_name"]=row["空間名稱"];room["room_type"]=row["空間類型"];room["include_in_area"]=bool(row["納入面積"])
else:st.info("尚無候選空間。")
st.markdown("### 比例尺與面積")
px=st.number_input("比例尺（px/m）",min_value=0.0,value=float(st.session_state.px_per_meter or 0))
if px>0:st.session_state.px_per_meter=px
load_per_ping=st.selectbox("每坪建議負荷值（kcal/h·坪）",LOAD_OPTIONS,index=4)
rows=[]
for room in st.session_state.rooms:
    ap=polygon_area_px2(room["points"]);am=pixel_area_to_m2(ap,st.session_state.px_per_meter);load=cooling_load(am,load_per_ping);inc=room.get("include_in_area",True)
    rows.append({"編號":room.get("room_id",""),"空間名稱":room.get("room_name",""),"納入面積":inc,"面積(px²)":round(ap,1),"面積(m²)":round(am,2) if inc and am is not None else None,"面積(坪)":round(load["ping"],2) if inc and load["ping"] is not None else None,"需求冷房能力(kcal/h)":round(load["kcal_h"]) if inc and load["kcal_h"] is not None else None})
area_df=pd.DataFrame(rows)
if not area_df.empty:st.dataframe(area_df,hide_index=True,use_container_width=True)
st.markdown("### 匯出")
e1,e2,e3=st.columns(3)
with e1:st.download_button("下載框線 JSON",json.dumps(st.session_state.rooms,ensure_ascii=False,indent=2).encode("utf-8"),f"{Path(uploaded.name).stem}_框線.json","application/json",use_container_width=True)
with e2:st.download_button("下載面積 CSV",area_df.to_csv(index=False).encode("utf-8-sig"),f"{Path(uploaded.name).stem}_面積.csv","text/csv",disabled=area_df.empty,use_container_width=True)
with e3:st.download_button("下載含底圖框面積 PDF",export_pdf(display_image,st.session_state.rooms,st.session_state.px_per_meter),f"{Path(uploaded.name).stem}_含底圖框面積.pdf","application/pdf",use_container_width=True)
