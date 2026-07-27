from __future__ import annotations
import io,json,re
from copy import deepcopy
from pathlib import Path
import fitz,gspread,pandas as pd,streamlit as st
from google.oauth2.service_account import Credentials
from PIL import Image,ImageDraw
from streamlit_drawable_canvas import st_canvas
from floorplan_detector import DetectorConfig,crop_to_main_floorplan,detect_room_polygons
from geometry_utils import cooling_load,fabric_line_endpoints,fabric_object_points,is_area_object,pixel_area_to_m2,polygon_area_px2,polygon_to_fabric_path,px_per_meter_from_line
from openai_reviewer import review_room_candidates

st.set_page_config(page_title='平面圖空調設備選型',page_icon='❄️',layout='wide')
COLORS=['#FF6347','#3B82F6','#22C55E','#F59E0B','#A855F7','#06B6D4'];LOAD_OPTIONS=list(range(400,1300,100))
for k,v in {'file_key':None,'drawing':{'version':'4.4.0','objects':[]},'canvas_version':0,'px_per_meter':None,'review':None,'equipment_table':None}.items():
    if k not in st.session_state:st.session_state[k]=v

@st.cache_data(show_spinner=False)
def pdf_page(data,page_index,dpi):
    doc=fitz.open(stream=data,filetype='pdf');page=doc.load_page(page_index);text=page.get_text();scale=None
    for p in [r'1\s*[:：]\s*(\d+)',r'1\s*/\s*(\d+)']:
        m=re.search(p,text)
        if m and 10<=int(m.group(1))<=5000:scale=int(m.group(1));break
    pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False);img=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);doc.close();return img,scale
@st.cache_data(show_spinner=False)
def image_file(data):return Image.open(io.BytesIO(data)).convert('RGB')
def resize(img,w):
    s=min(1,w/img.width);return (img.copy(),1) if s==1 else (img.resize((round(img.width*s),round(img.height*s)),Image.Resampling.LANCZOS),s)
def safe_bg(img):
    b=io.BytesIO();img.convert('RGBA').save(b,format='PNG');b.seek(0);return Image.open(b).convert('RGBA')
def objects():return st.session_state.drawing.get('objects',[])
def records():
    out=[];n=0
    for oi,o in enumerate(objects()):
        if is_area_object(o):
            n+=1;out.append({'room_id':o.get('room_id') or f'R{n:02d}','object_index':oi,'points':fabric_object_points(o),'color':o.get('stroke',COLORS[(n-1)%len(COLORS)]),'source':o.get('source','manual')})
    return out

def _col(r,i):return r[i].strip() if len(r)>i and r[i] else ''
@st.cache_data(show_spinner=False,ttl=300)
def equipment_data():
    try:
        creds=Credentials.from_service_account_info(dict(st.secrets['gcp_service_account']),scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']);gc=gspread.authorize(creds)
        sid=st.secrets.get('EQUIPMENT_SHEET_ID','1hEt4uxBABBicxIMJuR57lMiigQYF02CQHZfB-Nc6vjo');vals=gc.open_by_key(sid).get_worksheet(0).get_all_values();look={}
        for r in vals[2:]:
            indoor=_col(r,3) or _col(r,35)
            if indoor:look[indoor]={'類型':_col(r,1),'室外機':_col(r,2),'室內機冷房能力':_col(r,16)}
        return sorted(look),look
    except Exception:return [],{}
def export_pdf(img,recs,ppm):
    out=img.convert('RGB').copy();d=ImageDraw.Draw(out)
    for r in recs:
        pts=[(round(x),round(y)) for x,y in r['points']]
        if len(pts)<3:continue
        d.line(pts+[pts[0]],fill=r['color'],width=4);cx=round(sum(x for x,_ in pts)/len(pts));cy=round(sum(y for _,y in pts)/len(pts));a=pixel_area_to_m2(polygon_area_px2(r['points']),ppm);lab=r['room_id'] if a is None else f"{r['room_id']} {a:.2f}m2";d.text((cx,cy),lab,fill=r['color'])
    b=io.BytesIO();out.save(b,format='PDF',resolution=200);return b.getvalue()

st.markdown('## ❄️ 平面圖空調設備選型')
uploaded=st.file_uploader('上傳平面圖 PDF／PNG／JPG',type=['pdf','png','jpg','jpeg'])
if uploaded is None:st.info('請先上傳平面圖。');st.stop()
data=uploaded.getvalue();is_pdf=uploaded.name.lower().endswith('.pdf');page_index=0
if is_pdf:
    doc=fitz.open(stream=data,filetype='pdf');pc=doc.page_count;doc.close();page_index=st.selectbox('PDF頁面',range(pc),format_func=lambda i:f'第{i+1}頁')
with st.sidebar:
    st.header('圖面');dpi=st.slider('PDF解析度DPI',120,300,180,20);dw=st.slider('工作區寬度',700,1500,1100,50);crop=st.checkbox('自動裁切主要平面圖',True)
    st.header('自動辨識');mina=st.number_input('最小空間像素面積',1000,500000,7000,1000);maxr=st.slider('最大單一空間占比',.1,.9,.55,.05);wl=st.slider('牆線最短長度',8,100,24,2);wt=st.slider('牆線加粗',1,9,3);dg=st.slider('門洞封閉距離',5,100,28);eps=st.slider('框線簡化程度',.001,.03,.006,.001)
    st.header('框線');tool=st.radio('工具',['選取／拖曳','多邊形','四角形','校正線']);new_color=st.color_picker('新增框線顏色','#FF6347');sw=st.slider('框線粗細',1,8,3)
key=f'{uploaded.name}:{len(data)}:{hash(data)}:{page_index}:{dpi}:{dw}:{crop}'
if st.session_state.file_key!=key:
    st.session_state.file_key=key;st.session_state.drawing={'version':'4.4.0','objects':[]};st.session_state.canvas_version+=1;st.session_state.px_per_meter=None;st.session_state.review=None
img,auto_scale=pdf_page(data,page_index,dpi) if is_pdf else (image_file(data),None)
if crop:img=crop_to_main_floorplan(img)
img,_=resize(img,dw);img=safe_bg(img)

b1,b2,b3=st.columns(3)
with b1:
    if st.button('自動辨識並框面積',type='primary',use_container_width=True):
        cfg=DetectorConfig(wall_line_length=wl,wall_thickness=wt,door_gap_px=dg,min_room_area_px=mina,max_room_area_ratio=maxr,polygon_epsilon_ratio=eps);polys,_=detect_room_polygons(img.convert('RGB'),cfg);objs=[]
        for i,p in enumerate(polys,1):
            o=polygon_to_fabric_path(p,COLORS[(i-1)%len(COLORS)],sw,f'R{i:02d}','auto')
            if o:objs.append(o)
        st.session_state.drawing={'version':'4.4.0','objects':objs};st.session_state.canvas_version+=1;st.rerun()
with b2:
    if st.button('清空全部框線',use_container_width=True):st.session_state.drawing={'version':'4.4.0','objects':[]};st.session_state.canvas_version+=1;st.rerun()
with b3:debug=st.checkbox('顯示辨識中間結果')
if debug:
    cfg=DetectorConfig(wall_line_length=wl,wall_thickness=wt,door_gap_px=dg,min_room_area_px=mina,max_room_area_ratio=maxr,polygon_epsilon_ratio=eps);_,m=detect_room_polygons(img.convert('RGB'),cfg);c1,c2,c3=st.columns(3);c1.image(m['ink'],caption='線稿二值化');c2.image(m['walls'],caption='重建牆線');c3.image(m['interiors'],caption='封閉空間')
mode={'選取／拖曳':'transform','多邊形':'polygon','四角形':'rect','校正線':'line'}[tool]
st.caption('多邊形逐點點擊、雙擊完成；四角形拖曳繪製；選取模式可拖曳、拉伸與刪除。')
res=st_canvas(fill_color='rgba(0,0,0,0)',stroke_width=sw,stroke_color=new_color,background_image=img,update_streamlit=True,height=img.height,width=img.width,drawing_mode=mode,initial_drawing=st.session_state.drawing,display_toolbar=True,key=f"canvas_{st.session_state.canvas_version}")
if res.json_data is not None:st.session_state.drawing=deepcopy(res.json_data)
recs=records()

st.markdown('### 比例尺校正');lines=[o for o in objects() if o.get('type')=='line'];c1,c2,c3=st.columns(3)
with c1:actual_cm=st.number_input('最新校正線實際長度(cm)',1.0,value=1000.0)
with c2:
    if st.button('套用最新校正線',disabled=not lines,use_container_width=True):
        ep=fabric_line_endpoints(lines[-1]);st.session_state.px_per_meter=px_per_meter_from_line(ep[0],ep[1],actual_cm/100);st.rerun()
with c3:
    manual=st.number_input('或直接輸入px/m',0.0,value=float(st.session_state.px_per_meter or 0));
    if manual>0:st.session_state.px_per_meter=manual
if st.session_state.px_per_meter:st.success(f"目前比例尺：{st.session_state.px_per_meter:.3f} px/m")
elif auto_scale:st.warning(f'偵測到圖面比例1:{auto_scale}，仍建議用尺寸線校正。')
else:st.warning('尚未校正比例尺。')

st.markdown('### 框線管理')
if recs:
    sel=st.multiselect('選擇空間',range(len(recs)),format_func=lambda i:f"{recs[i]['room_id']}｜{recs[i]['source']}｜{recs[i]['color']}");m1,m2,m3=st.columns(3)
    with m1:
        if st.button('刪除選取空間',disabled=not sel,use_container_width=True):
            dels={recs[i]['object_index'] for i in sel};st.session_state.drawing['objects']=[o for i,o in enumerate(objects()) if i not in dels];st.session_state.canvas_version+=1;st.rerun()
    with m2:rc=st.color_picker('選取空間的新顏色','#3B82F6')
    with m3:
        if st.button('套用顏色',disabled=not sel,use_container_width=True):
            for i in sel:st.session_state.drawing['objects'][recs[i]['object_index']]['stroke']=rc
            st.session_state.canvas_version+=1;st.rerun()

st.markdown('### OpenAI圖面複核');api=st.secrets.get('OPENAI_API_KEY','');model=st.text_input('OpenAI視覺模型','gpt-4.1-mini')
if st.button('請OpenAI檢查候選框',disabled=not recs or not api):
    try:
        with st.spinner('OpenAI正在檢查…'):st.session_state.review=review_room_candidates(api,img.convert('RGB'),recs,model)
    except Exception as e:st.error(f'OpenAI複核失敗：{e}')
if not api:st.caption('尚未設定OPENAI_API_KEY；不影響OpenCV框選。')
if st.session_state.review:
    st.dataframe(pd.DataFrame(st.session_state.review.get('rooms',[])),use_container_width=True,hide_index=True)
    if st.session_state.review.get('missing_spaces'):st.dataframe(pd.DataFrame(st.session_state.review['missing_spaces']),use_container_width=True,hide_index=True)
    st.info(st.session_state.review.get('overall_note',''))

st.markdown('### 面積與空調負荷');load=st.selectbox('每坪建議負荷值(kcal/h·坪)',LOAD_OPTIONS,index=4);rows=[]
for r in recs:
    ap=polygon_area_px2(r['points']);am=pixel_area_to_m2(ap,st.session_state.px_per_meter);cl=cooling_load(am,load);rows.append({'編號':r['room_id'],'空間名稱':'','面積(px²)':round(ap,1),'面積(m²)':round(am,2) if am is not None else None,'面積(坪)':round(cl['ping'],2) if cl['ping'] is not None else None,'每坪建議負荷值':load,'需求冷房能力(kcal/h)':round(cl['kcal_h']) if cl['kcal_h'] is not None else None,'需求冷房能力(kW)':round(cl['kw'],2) if cl['kw'] is not None else None,'顏色':r['color']})
adf=pd.DataFrame(rows);st.dataframe(adf,use_container_width=True,hide_index=True) if not adf.empty else st.info('尚無空間資料。')

st.markdown('### 空調設備選型');models,lookup=equipment_data();eq=[]
for r in rows:
    prev=next((x for x in (st.session_state.equipment_table or []) if x.get('編號')==r['編號']),{});indoor=prev.get('室內機','');info=lookup.get(indoor,{})
    eq.append({'編號':r['編號'],'空間名稱':prev.get('空間名稱',''),'面積(m²)':r['面積(m²)'] or 0,'每坪建議負荷值':prev.get('每坪建議負荷值',load),'需求冷房能力':r['需求冷房能力(kcal/h)'] or 0,'室內機':indoor,'類型':info.get('類型',''),'室內機冷房能力':info.get('室內機冷房能力',''),'室外機':info.get('室外機',''),'連結率':prev.get('連結率','')})
ed=pd.DataFrame(eq);edited=st.data_editor(ed,num_rows='dynamic',use_container_width=True,column_config={'編號':st.column_config.TextColumn(disabled=True),'面積(m²)':st.column_config.NumberColumn(disabled=True),'每坪建議負荷值':st.column_config.SelectboxColumn(options=LOAD_OPTIONS),'需求冷房能力':st.column_config.NumberColumn(disabled=True),'室內機':st.column_config.SelectboxColumn(options=models or ['']),'類型':st.column_config.TextColumn(disabled=True),'室內機冷房能力':st.column_config.TextColumn(disabled=True),'室外機':st.column_config.TextColumn(disabled=True)},key='equipment_editor');st.session_state.equipment_table=edited.to_dict('records')

st.markdown('### 匯出');e1,e2,e3=st.columns(3)
with e1:st.download_button('下載框線JSON',json.dumps(st.session_state.drawing,ensure_ascii=False,indent=2).encode(),f'{Path(uploaded.name).stem}_框線.json','application/json',use_container_width=True)
with e2:st.download_button('下載面積CSV',adf.to_csv(index=False).encode('utf-8-sig'),f'{Path(uploaded.name).stem}_面積.csv','text/csv',disabled=adf.empty,use_container_width=True)
with e3:st.download_button('下載框選PDF',export_pdf(img,recs,st.session_state.px_per_meter),f'{Path(uploaded.name).stem}_框面積.pdf','application/pdf',use_container_width=True)
