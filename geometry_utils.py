from __future__ import annotations
import math
from typing import Any
from shapely.geometry import Polygon


def polygon_area_px2(points):
    pts=[(float(x),float(y)) for x,y in points]
    if len(pts)<3:return 0.0
    p=Polygon(pts)
    if not p.is_valid:p=p.buffer(0)
    return 0.0 if p.is_empty else float(p.area)


def pixel_area_to_m2(area_px2, px_per_meter):
    return None if not px_per_meter or px_per_meter<=0 else area_px2/(px_per_meter**2)


def px_per_meter_from_line(p1,p2,actual_m):
    if actual_m<=0: raise ValueError('實際長度必須大於0')
    px=math.dist(p1,p2)
    if px<=0: raise ValueError('校正線長度必須大於0')
    return px/actual_m


def cooling_load(area_m2, load_per_ping):
    if area_m2 is None:return {'ping':None,'kcal_h':None,'kw':None,'btu_h':None}
    ping=area_m2/3.3058
    kcal=ping*load_per_ping
    return {'ping':ping,'kcal_h':kcal,'kw':kcal*1.163/1000,'btu_h':kcal*3.96832}


def _tp(x,y,obj):
    sx=float(obj.get('scaleX',1) or 1); sy=float(obj.get('scaleY',1) or 1)
    left=float(obj.get('left',0) or 0); top=float(obj.get('top',0) or 0)
    a=math.radians(float(obj.get('angle',0) or 0)); x*=sx; y*=sy
    if a:x,y=x*math.cos(a)-y*math.sin(a),x*math.sin(a)+y*math.cos(a)
    return x+left,y+top


def fabric_object_points(obj:dict[str,Any]):
    t=obj.get('type','')
    if t=='rect':
        w=float(obj.get('width',0) or 0); h=float(obj.get('height',0) or 0)
        return [_tp(0,0,obj),_tp(w,0,obj),_tp(w,h,obj),_tp(0,h,obj)]
    if t=='polygon':
        po=obj.get('pathOffset',{'x':0,'y':0}); ox=float(po.get('x',0)); oy=float(po.get('y',0))
        return [_tp(float(p['x'])-ox,float(p['y'])-oy,obj) for p in obj.get('points',[])]
    if t=='path':
        pts=[]
        for c in obj.get('path',[]):
            if c and str(c[0]).upper() in {'M','L'} and len(c)>=3: pts.append(_tp(float(c[1]),float(c[2]),obj))
        if len(pts)>1 and pts[0]==pts[-1]:pts.pop()
        return pts
    return []


def fabric_line_endpoints(obj):
    if obj.get('type')!='line':return None
    return _tp(float(obj.get('x1',0)),float(obj.get('y1',0)),obj),_tp(float(obj.get('x2',0)),float(obj.get('y2',0)),obj)


def is_area_object(obj):
    return obj.get('type') in {'rect','polygon','path'} and len(fabric_object_points(obj))>=3


def polygon_to_fabric_path(points,color='#ff4b4b',stroke_width=3,room_id=None,source='auto'):
    pts=[(float(x),float(y)) for x,y in points]
    if len(pts)<3:return {}
    path=[['M',pts[0][0],pts[0][1]]]+[['L',x,y] for x,y in pts[1:]]+[['L',pts[0][0],pts[0][1]]]
    return {'type':'path','version':'4.4.0','originX':'left','originY':'top','left':0,'top':0,'width':0,'height':0,
            'fill':'rgba(0,0,0,0)','stroke':color,'strokeWidth':stroke_width,'strokeLineCap':'round','strokeLineJoin':'round',
            'scaleX':1,'scaleY':1,'angle':0,'opacity':1,'visible':True,'selectable':True,'evented':True,
            'room_id':room_id,'source':source,'path':path}
