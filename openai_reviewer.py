from __future__ import annotations
import base64,io,json
from PIL import Image,ImageDraw
from openai import OpenAI

SCHEMA={'type':'object','properties':{
'rooms':{'type':'array','items':{'type':'object','properties':{
'room_id':{'type':'string'},'room_type':{'type':'string'},'confidence':{'type':'number'},
'action':{'type':'string','enum':['keep','remove','merge','review']},'merge_with':{'type':'array','items':{'type':'string'}},'reason':{'type':'string'}},
'required':['room_id','room_type','confidence','action','merge_with','reason'],'additionalProperties':False}},
'missing_spaces':{'type':'array','items':{'type':'object','properties':{'description':{'type':'string'},'approximate_location':{'type':'string'},'confidence':{'type':'number'}},'required':['description','approximate_location','confidence'],'additionalProperties':False}},
'overall_note':{'type':'string'}},'required':['rooms','missing_spaces','overall_note'],'additionalProperties':False}


def _url(img):
    b=io.BytesIO();img.convert('RGB').save(b,format='JPEG',quality=90)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()


def review_room_candidates(api_key,image,room_records,model='gpt-4.1-mini'):
    if not api_key:raise ValueError('尚未設定 OPENAI_API_KEY')
    out=image.convert('RGB').copy();d=ImageDraw.Draw(out)
    for i,r in enumerate(room_records,1):
        pts=[(round(x),round(y)) for x,y in r['points']]; color=r.get('color','#ff4b4b');rid=r.get('room_id',f'R{i:02d}')
        if len(pts)<3:continue
        d.line(pts+[pts[0]],fill=color,width=4);cx=round(sum(x for x,_ in pts)/len(pts));cy=round(sum(y for _,y in pts)/len(pts))
        d.rectangle((cx-25,cy-12,cx+25,cy+12),fill='white',outline=color,width=2);d.text((cx-19,cy-8),rid,fill=color)
    prompt='''你是建築平面圖候選空間品質檢查助手。檢查每個編號框是否為牆體圍合的獨立空間，推測空間類型，並給出 keep/remove/merge/review。檢查明顯漏框。只做語意檢查，不虛構尺寸，不宣稱 CAD 精度。門後玄關若仍在同一牆體圍合內，通常屬於同一空間。'''
    client=OpenAI(api_key=api_key)
    resp=client.responses.create(model=model,input=[{'role':'user','content':[{'type':'input_text','text':prompt},{'type':'input_image','image_url':_url(out),'detail':'high'}]}],text={'format':{'type':'json_schema','name':'floorplan_room_review','strict':True,'schema':SCHEMA}})
    return json.loads(resp.output_text)
