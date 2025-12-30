import os
from PIL import Image, ImageChops, ImageEnhance

in_path = "./out_png/page_001.png"          # 改成你的大图
out_dir = "tiles_out"
os.makedirs(out_dir, exist_ok=True)

def trim_white_border(im: Image.Image, tol=245):
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    # 仅保留明显非白内容
    diff = diff.point(lambda p: 0 if p > (255 - tol) else 255)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def tile(im: Image.Image, rows=3, cols=4, overlap=160):
    w, h = im.size
    tw = w // cols
    th = h // rows
    k = 0
    for r in range(rows):
        for c in range(cols):
            left   = max(0, c*tw - overlap)
            upper  = max(0, r*th - overlap)
            right  = min(w, (c+1)*tw + overlap)
            lower  = min(h, (r+1)*th + overlap)
            yield k, im.crop((left, upper, right, lower)), (left, upper, right, lower)
            k += 1

im = Image.open(in_path)
im = trim_white_border(im, tol=245)

# 可选：轻微增强，帮助读小字（别太猛）
im = ImageEnhance.Contrast(im).enhance(1.15)
im = ImageEnhance.Sharpness(im).enhance(1.2)

for k, t, box in tile(im, rows=3, cols=4, overlap=200):
    # 图纸线条适合 PNG；如果仍太大可改 JPEG(quality=92)
    t.save(os.path.join(out_dir, f"tile_{k:02d}_{box[0]}_{box[1]}_{box[2]}_{box[3]}.png"),
           "PNG", optimize=True)