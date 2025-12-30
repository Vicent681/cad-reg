import os
from pdf2image import convert_from_path
from PIL import Image, ImageChops

# pdf_path = "/Users/vincent/Desktop/1号楼墙柱平法施工图.pdf"
pdf_path = '/Users/vincent/Desktop/3、6#楼立面图20250528.pdf'

out_dir = "out_png"
os.makedirs(out_dir, exist_ok=True)

def trim_white_border(im: Image.Image, tol: int = 245) -> Image.Image:
    """
    裁掉四周白边/近白边。tol 越大裁得越激进(更容易把浅色线裁掉)，建议 240~250 试。
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    # 把“接近白色”的差异压到0，仅保留明显内容
    diff = diff.point(lambda p: 0 if p > (255 - tol) else 255)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

pages = convert_from_path(pdf_path, dpi=400)

for i, img in enumerate(pages, start=1):
    img = trim_white_border(img, tol=245)
    img.save(os.path.join(out_dir, f"page_{i:03d}.png"), "PNG", optimize=True)