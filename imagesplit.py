import os
import cv2
from collections import defaultdict

import numpy as np


# 局部合并函数（基于扩展相交）
def expand_rect(rect, d):
    x, y, w, h = rect
    return (x - d, y - d, w + 2 * d, h + 2 * d)

def rects_intersect(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def merge_two_rects(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x = min(x1, x2)
    y = min(y1, y2)
    xmax = max(x1 + w1, x2 + w2)
    ymax = max(y1 + h1, y2 + h2)
    return (x, y, xmax - x, ymax - y)

def merge_rects_in_list(rect_list, threshold):
    """迭代合并列表中的矩形，直到无法再合并"""
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(rect_list)
        expanded = [expand_rect(r, threshold) for r in rect_list]

        for i in range(len(rect_list)):
            if used[i]:
                continue
            current = rect_list[i]
            for j in range(i + 1, len(rect_list)):
                if not used[j] and rects_intersect(expanded[i], expanded[j]):
                    current = merge_two_rects(current, rect_list[j])
                    used[j] = True
                    changed = True
            merged.append(current)
        rect_list = merged
    return rect_list


def graph_split(img_path, tmp_path, merge_threshold=11.5, grid_size=200, max_area_riot=0.25, min_area=150):
    """
    参数：
    merge_threshold：连通域合并阈值
    grid_size：图像网格大小参数
    max_area_riot：最大连通域过滤参数
    min_area：输出结果最小矩形过滤参数
    """
    # 1. 读取图像 & 获取初始矩形
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # RETR_EXTERNAL：只检测最外层轮廓。
    # RETR_LIST：检测所有轮廓，不建立层级关系。
    # RETR_TREE：检测所有轮廓，并重建完整的嵌套层级结构。
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 连通域分析（8-连通）
    # num_labels, labels_im = cv2.connectedComponents(binary, connectivity=8)
    # print(f"检测到 {num_labels - 1} 个连通区域（含背景）")

    h_img, w_img = img.shape[:2]
    long_side = max(h_img, w_img)
    max_allowed_side = long_side * max_area_riot  # 长边的 1/4

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 最大连通域过滤，任一边超过原图长边的 1/4 则跳过
        if w > max_allowed_side or h > max_allowed_side:
            continue
        rects.append((x, y, w, h))
    # logging.info(f"过滤后保留 {len(rects)} 个矩形")
    # for label in range(1, num_labels):
    #     # 获取该标签的掩码
    #     mask = (labels_im == label).astype(np.uint8) * 255
    #     # 找最小外接矩形
    #     coords = cv2.findNonZero(mask)
    #     x, y, w, h = cv2.boundingRect(coords)
    #
    #     # 最大连通域过滤，任一边超过原图长边的 1/4 则跳过
    #     if w > max_allowed_side or h > max_allowed_side:
    #         continue
    #     rects.append((x, y, w, h))

    # 2. 构建网格索引：每个网格包含哪些矩形
    grid_map = defaultdict(list)

    for i, (x, y, w, h) in enumerate(rects):
        # 计算该矩形覆盖的网格范围（向上取整）
        gx_min = x // grid_size
        gy_min = y // grid_size
        gx_max = (x + w - 1) // grid_size
        gy_max = (y + h - 1) // grid_size

        # # 加入所有覆盖的网格
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                grid_map[(gx, gy)].append(i)

    # 3. 对每个网格内的矩形进行局部合并
    local_merged = set()  # 使用 set 避免重复（因为一个矩形可能在多个网格）

    for grid_key, indices in grid_map.items():
        if len(indices) <= 1:
            continue
        local_rects = [rects[i] for i in indices]
        merged_in_grid = merge_rects_in_list(local_rects, merge_threshold)
        for r in merged_in_grid:
            local_merged.add(r)

    local_merged = list(local_merged)

    # 4. 全局二次合并（处理跨网格未合并的情况）
    final_rects = merge_rects_in_list(local_merged, merge_threshold)
    final_rects = [
        (x, y, w, h) for (x, y, w, h) in final_rects
        if w >= min_area and h >= min_area
    ]
    # logging.info(f"最终合并后剩余 {len(final_rects)} 个矩形")
    # 5. 可视化结果
    # output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for (x, y, w, h) in final_rects:
    #     cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # cv2.imwrite('with_merged_boxes_optimized.png', output_img)

    # 遍历所有最终合并的矩形，裁剪并保存
    result = []
    os.makedirs(tmp_path, exist_ok=True)
    for idx, (x, y, w, h) in enumerate(final_rects):
        # 边界安全检查（防止越界）
        x = max(0, x)
        y = max(0, y)
        x2 = min(x + w, img.shape[1])
        y2 = min(y + h, img.shape[0])

        if x2 <= x or y2 <= y:
            continue  # 无效区域，跳过

        cropped = img[y:y2, x:x2]

        # 保存为 crop_000.png, crop_001.png, ...
        filename = f"crop_{idx:03d}.png"
        save_path = os.path.join(tmp_path, filename)
        cv2.imwrite(save_path, cropped)
        result.append(save_path)

    return result


if __name__ == "__main__":
    graph_split(
        r"./out_png/page_001.png",
        "./box_img",
    )

