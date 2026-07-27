from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np
from PIL import Image
from shapely.geometry import Polygon

@dataclass
class DetectorConfig:
    wall_line_length:int=24; wall_thickness:int=3; door_gap_px:int=28
    min_room_area_px:int=7000; max_room_area_ratio:float=.55
    polygon_epsilon_ratio:float=.006; adaptive_block_size:int=31; adaptive_c:int=11


def crop_to_main_floorplan(image:Image.Image,padding=25):
    g=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
    _,ink=cv2.threshold(g,245,255,cv2.THRESH_BINARY_INV)
    ink=cv2.dilate(ink,np.ones((5,5),np.uint8),iterations=2)
    n,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    if n<=1:return image
    best=None; score=-1
    for i in range(1,n):
        x,y,w,h,a=stats[i]; ba=w*h
        if ba and (a/ba)*a>score:score=(a/ba)*a;best=i
    if best is None:return image
    x,y,w,h,_=stats[best]; x0=max(0,x-padding);y0=max(0,y-padding);x1=min(image.width,x+w+padding);y1=min(image.height,y+h+padding)
    return image.crop((x0,y0,x1,y1))


def _remove_small(mask,min_area=30):
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8);out=np.zeros_like(mask)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=min_area:out[lab==i]=255
    return out


def _remove_outside(free):
    h,w=free.shape; f=free.copy(); fm=np.zeros((h+2,w+2),np.uint8)
    seeds=[]
    for x in range(0,w,max(1,w//150)):seeds += [(x,0),(x,h-1)]
    for y in range(0,h,max(1,h//150)):seeds += [(0,y),(w-1,y)]
    for x,y in seeds:
        if f[y,x]==255:cv2.floodFill(f,fm,(x,y),128)
    out=np.zeros_like(free);out[f==255]=255;return out


def detect_room_polygons(image:Image.Image,config=None):
    c=config or DetectorConfig(); rgb=np.array(image.convert('RGB')); g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    bs=c.adaptive_block_size if c.adaptive_block_size%2 else c.adaptive_block_size+1
    ink=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,c.adaptive_c)
    L=max(8,c.wall_line_length)
    hline=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(L,1)))
    vline=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,L)))
    seed=cv2.bitwise_or(hline,vline); near=cv2.dilate(seed,cv2.getStructuringElement(cv2.MORPH_RECT,(13,13)))
    walls=cv2.bitwise_or(cv2.bitwise_and(ink,near),seed)
    t=max(1,c.wall_thickness);walls=cv2.dilate(walls,cv2.getStructuringElement(cv2.MORPH_RECT,(t,t)))
    gap=max(3,c.door_gap_px)
    wh=cv2.morphologyEx(walls,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(gap,3)))
    wv=cv2.morphologyEx(walls,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,gap)))
    walls=_remove_small(cv2.bitwise_or(wh,wv))
    interiors=_remove_outside(cv2.bitwise_not(walls))
    n,lab,stats,_=cv2.connectedComponentsWithStats(interiors,8); H,W=interiors.shape; polys=[]
    for i in range(1,n):
        a=stats[i,cv2.CC_STAT_AREA];bw=stats[i,cv2.CC_STAT_WIDTH];bh=stats[i,cv2.CC_STAT_HEIGHT]
        if a<c.min_room_area_px or a>H*W*c.max_room_area_ratio or min(bw,bh)<25:continue
        comp=np.zeros_like(interiors);comp[lab==i]=255
        contours,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:continue
        cnt=max(contours,key=cv2.contourArea); peri=cv2.arcLength(cnt,True)
        ap=cv2.approxPolyDP(cnt,c.polygon_epsilon_ratio*peri,True)
        pts=[(float(p[0][0]),float(p[0][1])) for p in ap]
        if 3<=len(pts)<=60:
            p=Polygon(pts)
            if p.is_valid and p.area>=c.min_room_area_px:polys.append(pts)
    polys.sort(key=lambda pts:(min(y for _,y in pts),min(x for x,_ in pts)))
    return polys,{'ink':ink,'walls':walls,'interiors':interiors}
