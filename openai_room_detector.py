from __future__ import annotations
import base64, io, json, time
from dataclasses import dataclass
from typing import Any
from openai import OpenAI
from PIL import Image
from shapely.geometry import Polygon

GLOBAL_SCHEMA={"type":"object","properties":{"assessment":{"type":"object","properties":{"is_floor_plan":{"type":"boolean"},"quality":{"type":"string","enum":["good","usable","poor"]},"note":{"type":"string"}},"required":["is_floor_plan","quality","note"],"additionalProperties":False},"rooms":{"type":"array","items":{"type":"object","properties":{"room_name":{"type":"string"},"room_type":{"type":"string"},"include_in_area":{"type":"boolean"},"confidence":{"type":"number"},"bbox":{"type":"object","properties":{"x1":{"type":"integer","minimum":0,"maximum":1000},"y1":{"type":"integer","minimum":0,"maximum":1000},"x2":{"type":"integer","minimum":0,"maximum":1000},"y2":{"type":"integer","minimum":0,"maximum":1000}},"required":["x1","y1","x2","y2"],"additionalProperties":False},"reason":{"type":"string"}},"required":["room_name","room_type","include_in_area","confidence","bbox","reason"],"additionalProperties":False}}},"required":["assessment","rooms"],"additionalProperties":False}
LOCAL_SCHEMA={"type":"object","properties":{"valid_room":{"type":"boolean"},"confidence":{"type":"number"},"polygon":{"type":"array","minItems":4,"maxItems":20,"items":{"type":"object","properties":{"x":{"type":"integer","minimum":0,"maximum":1000},"y":{"type":"integer","minimum":0,"maximum":1000}},"required":["x","y"],"additionalProperties":False}},"note":{"type":"string"}},"required":["valid_room","confidence","polygon","note"],"additionalProperties":False}

@dataclass
class AIRoomDetectionOptions:
    minimum_confidence: float=0.45
    maximum_rooms: int=18
    context_ratio: float=0.20
    max_retries: int=2
    include_balcony: bool=True
    include_corridor: bool=True
    include_bathroom: bool=True
    include_stair: bool=False

def _data_url(image:Image.Image)->str:
    b=io.BytesIO();image.convert("RGB").save(b,format="JPEG",quality=92)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

def _ask(client,model,prompt,image,name,schema,retries):
    for attempt in range(retries+1):
        try:
            r=client.responses.create(model=model,input=[{"role":"user","content":[{"type":"input_text","text":prompt},{"type":"input_image","image_url":_data_url(image),"detail":"high"}]}],text={"format":{"type":"json_schema","name":name,"strict":True,"schema":schema}})
            return json.loads(r.output_text)
        except Exception:
            if attempt>=retries:raise
            time.sleep(1.5*(attempt+1))

def _bbox(b,w,h):
    x1=round(b["x1"]/1000*w);y1=round(b["y1"]/1000*h);x2=round(b["x2"]/1000*w);y2=round(b["y2"]/1000*h)
    x1,x2=sorted((max(0,x1),min(w,x2)));y1,y2=sorted((max(0,y1),min(h,y2)))
    return x1,y1,max(2,x2-x1),max(2,y2-y1)

def _expand(box,w,h,r):
    x,y,bw,bh=box;mx=round(bw*r);my=round(bh*r);x0=max(0,x-mx);y0=max(0,y-my);x1=min(w,x+bw+mx);y1=min(h,y+bh+my)
    return x0,y0,x1-x0,y1-y0

def _repair(points,w,h):
    if len(points)<3:return []
    p=Polygon([(min(max(float(x),0),w-1),min(max(float(y),0),h-1)) for x,y in points])
    if not p.is_valid:p=p.buffer(0)
    if p.is_empty:return []
    if p.geom_type=="MultiPolygon":p=max(p.geoms,key=lambda g:g.area)
    p=p.simplify(max(w,h)*0.001,preserve_topology=True)
    return [(float(x),float(y)) for x,y in list(p.exterior.coords)[:-1]]

def _include(room,opt):
    t=(room.get("room_name","")+" "+room.get("room_type","")).lower()
    if any(s in t for s in ["樓梯","梯間","stair"]):return opt.include_stair
    if any(s in t for s in ["陽台","露台","balcony"]):return opt.include_balcony
    if any(s in t for s in ["走道","走廊","玄關","corridor"]):return opt.include_corridor
    if any(s in t for s in ["衛浴","浴室","廁所","bathroom","toilet"]):return opt.include_bathroom
    return bool(room.get("include_in_area",True))

def _iou(a,b):
    u=a.union(b).area
    return 0 if u<=0 else float(a.intersection(b).area/u)

def detect_rooms_with_openai(api_key,image,model="gpt-4.1",options=None):
    if not api_key:raise ValueError("尚未設定 OPENAI_API_KEY。")
    opt=options or AIRoomDetectionOptions();src=image.convert("RGB");client=OpenAI(api_key=api_key)
    gp=f"""完整判讀這張建築平面圖。第一階段只回傳每個實際空間的名稱、用途與大致 bbox。bbox 使用 0～1000 座標；一個 bbox 只包含一個空間；忽略尺寸線、文字、家具、門片弧線、窗框內線、樓梯踏階與基地線；開放式客餐廳沒有實牆時視為同一空間；不確定就降低 confidence。最多 {opt.maximum_rooms} 個空間。"""
    g=_ask(client,model,gp,src,"floorplan_global_rooms",GLOBAL_SCHEMA,opt.max_retries)
    accepted=[];rejected=[];full=src.width*src.height
    for room in g.get("rooms",[]):
        gc=float(room.get("confidence",0))
        if gc<opt.minimum_confidence:
            rejected.append({"room_name":room.get("room_name","未命名"),"confidence":gc,"rejected_reason":"全圖信心不足"});continue
        box=_bbox(room["bbox"],src.width,src.height);x,y,w,h=_expand(box,src.width,src.height,opt.context_ratio);crop=src.crop((x,y,x+w,y+h))
        lp=f"""這是「{room.get('room_name','未命名空間')}」附近局部圖。只框指定空間的內牆完成面。polygon 使用局部圖 0～1000 座標；不得包含相鄰空間；門洞沿牆面方向補成邊界；忽略家具、尺寸線、文字、門片弧線、樓梯踏階與窗框內線；L 型或凹型要保留轉折。若無法可靠判斷，valid_room=false。"""
        l=_ask(client,model,lp,crop,"floorplan_local_polygon",LOCAL_SCHEMA,opt.max_retries);lc=float(l.get("confidence",0));conf=min(gc,lc)
        if not l.get("valid_room",False):
            rejected.append({"room_name":room.get("room_name","未命名"),"confidence":conf,"rejected_reason":"局部無法確認"});continue
        pts=_repair([(x+p["x"]/1000*w,y+p["y"]/1000*h) for p in l.get("polygon",[])],src.width,src.height);area=float(Polygon(pts).area) if len(pts)>=3 else 0
        reasons=[]
        if conf<opt.minimum_confidence:reasons.append("局部信心不足")
        if len(pts)<3:reasons.append("多邊形無效")
        if area<full*0.0007:reasons.append("面積過小")
        if area>full*0.24:reasons.append("面積過大")
        rec={"room_id":"","room_name":room.get("room_name") or "未命名空間","room_type":room.get("room_type") or "無法判斷","include_in_area":_include(room,opt),"confidence":conf,"points":pts,"area_px2":area,"reason":room.get("reason","")+"；"+l.get("note",""),"source":"gpt_full_and_local"}
        if reasons:rec["rejected_reason"]="、".join(reasons);rejected.append(rec)
        else:accepted.append(rec)
    out=[]
    for cand in sorted(accepted,key=lambda r:r["confidence"],reverse=True):
        p=Polygon(cand["points"])
        if any(_iou(p,Polygon(o["points"]))>=0.48 for o in out):continue
        out.append(cand)
    out.sort(key=lambda r:(min(y for _,y in r["points"]),min(x for x,_ in r["points"])))
    for i,r in enumerate(out,1):r["room_id"]=f"R{i:02d}"
    return {"rooms":out,"rejected_rooms":rejected,"image_assessment":{"is_floor_plan":g.get("assessment",{}).get("is_floor_plan",True),"quality":g.get("assessment",{}).get("quality","usable"),"note":g.get("assessment",{}).get("note","")},"overall_note":"每張圖先由 GPT 完整判讀，再逐房間放大精修。","model":model,"image_width":src.width,"image_height":src.height}
