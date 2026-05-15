"""Generative-art tools beyond classic fractals."""

from __future__ import annotations

import math, random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from plugins.BaseTool import ToolResult
from plugins.tools.tool_fractal_suite import PALETTES, _Base, _clamp, _color, _pick, _save


def _rgb(a, pal, seed, mask=None):
    a = np.nan_to_num(a.astype("float32")); a = (a - a.min()) / ((a.max() - a.min()) or 1)
    base = {"plasma": 4.9, "laser": 3.3, "sunset": .2, "ice": 3.7, "toxic": 1.5, "royal": 4.2}.get(pal, 4.9)
    out = np.zeros((*a.shape, 3), dtype=np.uint8)
    for i, off in enumerate((0, 2.1, 4.2)):
        out[..., i] = (255 * (.5 + .5 * np.sin(base + off + a * (6 + i))) * (.2 + .9 * a)).clip(0, 255)
    if mask is not None: out[~mask] = (3, 4, 9)
    return Image.fromarray(out, "RGB")


class RenderStrangeAttractor(_Base):
    name = "render_strange_attractor"
    description = "Render a high-density strange attractor point cloud: Clifford, De Jong, or Hopalong. Chaotic, elegant, and very r/generative."
    parameters = {**_Base.parameters, "properties": {**_Base.parameters["properties"], "preset": {"type": "string", "enum": ["clifford", "dejong", "hopalong"]}}}
    def run(self, context, **kw):
        w,h,d,p,s = self.args(kw); preset = _pick(kw.get("preset"), {"clifford","dejong","hopalong"}, "clifford"); r=random.Random(s); n=d*4200
        img=Image.new("RGB",(w,h),(2,3,8)); px=img.load(); x=y=0.; pts=[]
        a,b,c,e = [r.uniform(-2.4,2.4) for _ in range(4)]
        for i in range(n):
            if preset=="dejong": x,y = math.sin(a*y)-math.cos(b*x), math.sin(c*x)-math.cos(e*y)
            elif preset=="hopalong": x,y = y - math.copysign(math.sqrt(abs(b*x-c)), x), a-x
            else: x,y = math.sin(a*y)+c*math.cos(a*x), math.sin(b*x)+e*math.cos(b*y)
            if i>200: pts.append((x,y))
        xs,ys=zip(*pts); mnx,mxx,mny,mxy=min(xs),max(xs),min(ys),max(ys)
        for i,(x,y) in enumerate(pts):
            X=int((x-mnx)/(mxx-mnx or 1)*(w-1)); Y=int((y-mny)/(mxy-mny or 1)*(h-1))
            if 0<=X<w and 0<=Y<h: px[X,Y]=_color(i/len(pts),p,s,1.25)
        return _save("strange_attractor", img.filter(ImageFilter.SMOOTH_MORE), {"preset": preset, "seed": s, "palette": p}, context)


class RenderFlowField(_Base):
    name = "render_flow_field"
    description = "Render sweeping particle trails through a procedural vector field. Great for silk, smoke, magnetic-field, and ink-stream looks."
    def run(self, context, **kw):
        w,h,d,p,s=self.args(kw); r=random.Random(s); img=Image.new("RGB",(w,h),(2,3,8)); draw=ImageDraw.Draw(img,"RGBA")
        for j in range(900):
            x,y=r.random()*w,r.random()*h; pts=[]
            for i in range(90):
                a=math.sin(x*.012+r.random()*.02+s)+math.cos(y*.015+j*.002); x+=math.cos(a*3)*3.2; y+=math.sin(a*3)*3.2
                if not (0<=x<w and 0<=y<h): break
                pts.append((x,y))
            if len(pts)>2: draw.line(pts, fill=(*_color(j/900,p,s,1.1), 42), width=r.choice([1,1,2]))
        return _save("flow_field", img.filter(ImageFilter.GaussianBlur(.25)), {"seed": s, "palette": p}, context)


class RenderCellularAutomata(_Base):
    name = "render_cellular_automata"
    description = "Render cellular automata art: Conway-like growth, maze rules, or Wolfram elementary automata stacked into a gorgeous history image."
    parameters = {**_Base.parameters, "properties": {**_Base.parameters["properties"], "rule": {"type": "string", "enum": ["life", "maze", "rule30", "rule110"]}}}
    def run(self, context, **kw):
        w,h,d,p,s=self.args(kw); rule=_pick(kw.get("rule"),{"life","maze","rule30","rule110"},"life"); rng=np.random.default_rng(s)
        if rule.startswith("rule"):
            code=int(rule[4:]); row=rng.integers(0,2,w,dtype=np.uint8); canvas=np.zeros((h,w),dtype=np.float32)
            for y in range(h):
                canvas[y]=row; idx=(np.roll(row,1)<<2)|(row<<1)|np.roll(row,-1); row=((code>>idx)&1).astype(np.uint8)
        else:
            grid=rng.random((h,w))>.72; canvas=np.zeros((h,w),dtype=np.float32)
            for i in range(min(d*3,700)):
                n=sum(np.roll(np.roll(grid,a,0),b,1) for a in (-1,0,1) for b in (-1,0,1) if a or b)
                grid = (n==3) | (grid & (n==2)) if rule=="life" else (n==3) | (grid & (n>=1) & (n<=5))
                canvas += grid * (i / min(d*3,700))
        return _save("cellular_automata", _rgb(canvas,p,s,canvas>0), {"rule": rule, "seed": s, "palette": p}, context)


class RenderReactionDiffusion(_Base):
    name = "render_reaction_diffusion"
    description = "Render Turing/Gray-Scott reaction-diffusion textures: coral, leopard spots, alien chemistry, and living marble."
    def run(self, context, **kw):
        w,h,d,p,s=self.args(kw); w,h=min(w,1000),min(h,800); rng=np.random.default_rng(s); A=np.ones((h,w),np.float32); B=np.zeros((h,w),np.float32)
        for _ in range(18):
            x,y=rng.integers(0,w),rng.integers(0,h); B[max(0,y-16):y+16,max(0,x-16):x+16]=rng.random()
        F,K=.035+rng.random()*.035,.055+rng.random()*.01
        for _ in range(min(d,260)):
            LA=sum(np.roll(np.roll(A,i,0),j,1)*v for i,j,v in [(0,1,.2),(1,0,.2),(-1,0,.2),(0,-1,.2),(1,1,.05),(1,-1,.05),(-1,1,.05),(-1,-1,.05),(0,0,-1)])
            LB=sum(np.roll(np.roll(B,i,0),j,1)*v for i,j,v in [(0,1,.2),(1,0,.2),(-1,0,.2),(0,-1,.2),(1,1,.05),(1,-1,.05),(-1,1,.05),(-1,-1,.05),(0,0,-1)])
            R=A*B*B; A += 1.0*LA - R + F*(1-A); B += .5*LB + R - (K+F)*B
        return _save("reaction_diffusion", _rgb(B,p,s), {"seed": s, "palette": p}, context)


class RenderLeniaLike(_Base):
    name = "render_lenia_like"
    description = "Render a continuous cellular-automata/Lenia-inspired organism field: soft blobs, membranes, and luminous artificial life."
    def run(self, context, **kw):
        w,h,d,p,s=self.args(kw); w,h=min(w,900),min(h,700); rng=np.random.default_rng(s); a=rng.random((h,w),dtype=np.float32)
        for _ in range(min(d*2,320)):
            n=(np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1)+np.roll(np.roll(a,1,0),1,1)+np.roll(np.roll(a,1,0),-1,1)+np.roll(np.roll(a,-1,0),1,1)+np.roll(np.roll(a,-1,0),-1,1))/8
            a=np.clip(a + .16*(np.exp(-((n-.38)**2)/.018)*2-1),0,1)
        return _save("lenia_like", _rgb(a,p,s), {"seed": s, "palette": p}, context)


class RenderPixelMosaic(_Base):
    name = "render_pixel_mosaic"
    description = "Render r/place-inspired pixel mosaic art: collaborative-looking tiles, glyphs, noisy edits, and color territories."
    def run(self, context, **kw):
        w,h,d,p,s=self.args(kw); r=random.Random(s); cell=max(6,min(18,w//90)); img=Image.new("RGB",(w,h),(3,4,9)); draw=ImageDraw.Draw(img)
        colors=[_color(i/9,p,s,1.25) for i in range(10)]
        for y in range(0,h,cell):
            for x in range(0,w,cell):
                v=math.sin(x*.025+s)+math.cos(y*.022)+math.sin((x+y)*.014); c=colors[int(abs(v*3+r.random()*2))%len(colors)]
                draw.rectangle((x,y,x+cell-1,y+cell-1),fill=c)
        for _ in range(d*8):
            x,y=r.randrange(0,w,cell),r.randrange(0,h,cell); draw.rectangle((x,y,x+cell*r.randint(1,5),y+cell*r.randint(1,5)),fill=r.choice(colors))
        return _save("pixel_mosaic", img.resize((w,h), Image.Resampling.NEAREST), {"seed": s, "palette": p}, context)
