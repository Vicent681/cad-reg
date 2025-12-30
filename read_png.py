import cv2

# img_path = "./out_png/page_001.png"
img_path = "/Users/vincent/Project/cad-reg/roi_outputs/page_001_det.png"
# 改成你的png路径
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(img_path)

win = "img"
overlay = img.copy()
last_xy = (-1, -1)

def on_mouse(event, x, y, flags, param):
    global overlay, last_xy
    if event == cv2.EVENT_MOUSEMOVE:
        last_xy = (x, y)
        overlay = img.copy()
        cv2.putText(
            overlay, f"x={x}, y={y}",
            (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 6,
            (255, 0, 0), 2, cv2.LINE_AA
        )

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win, on_mouse)

while True:
    cv2.imshow(win, overlay)
    key = cv2.waitKey(10) & 0xFF
    if key in (27, ord("q")):  # ESC 或 q 退出
        break

cv2.destroyAllWindows()