#!/usr/bin/env python3
"""Split high-resolution CAD drawings into overlapping image tiles."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageChops, ImageEnhance


@dataclass
class TileInfo:
    tile_index: int
    path: Path
    bbox: Tuple[int, int, int, int]
    row: int
    col: int
    source_image: Path
    is_full_image: bool = False


def trim_white_border(im: Image.Image, tol: int = 245) -> Image.Image:
    """Trim near-white borders."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    diff = diff.point(lambda p: 0 if p > (255 - tol) else 255)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


def enhance_for_legibility(im: Image.Image, contrast: float = 1.15, sharpness: float = 1.2) -> Image.Image:
    """Apply slight contrast/sharpness boost to help small annotations stand out."""
    enhanced = ImageEnhance.Contrast(im).enhance(contrast)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
    return enhanced


def _determine_grid_layout(
    width: int,
    height: int,
    *,
    max_tile_dim: int = 1200,
    max_tiles: int = 4,
    requested_rows: int = 3,
    requested_cols: int = 4,
) -> tuple[int, int]:
    if width <= max_tile_dim and height <= max_tile_dim:
        return 1, 1

    min_rows_needed = max(1, math.ceil(height / max_tile_dim))
    min_cols_needed = max(1, math.ceil(width / max_tile_dim))

    max_row = min(max_tiles, max(requested_rows, min_rows_needed))
    max_col = min(max_tiles, max(requested_cols, min_cols_needed))

    candidates: List[tuple[int, int, int]] = []
    for rows in range(1, max_row + 1):
        for cols in range(1, max_col + 1):
            if rows * cols > max_tiles:
                continue
            if height / rows <= max_tile_dim and width / cols <= max_tile_dim:
                candidates.append((rows * cols, rows, cols))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        _, rows, cols = candidates[0]
        return rows, cols

    # 如果无法在 max_tiles 内满足 1200 限制，退回到最小需要切分数
    return min_rows_needed, min_cols_needed


def split_image_into_tiles(
    image_path: Path,
    output_dir: Path,
    *,
    rows: int = 3,
    cols: int = 4,
    overlap: int = 200,
    trim: bool = False,
    trim_tol: int = 245,
    enhance: bool = False,
    contrast: float = 1.15,
    sharpness: float = 1.2,
    include_full_image: bool = False,
) -> List[TileInfo]:
    """Split a large CAD image into overlapping tiles."""
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path)
    if trim:
        image = trim_white_border(image, tol=trim_tol)
    if enhance:
        image = enhance_for_legibility(image, contrast=contrast, sharpness=sharpness)

    width, height = image.size
    effective_rows, effective_cols = _determine_grid_layout(
        width,
        height,
        max_tile_dim=1200,
        max_tiles=4,
        requested_rows=rows,
        requested_cols=cols,
    )
    tile_width = width // effective_cols
    tile_height = height // effective_rows
    tiles: List[TileInfo] = []

    if include_full_image:
        full_path = output_dir / "tile_full.png"
        image.save(full_path, "PNG", optimize=True)
        tiles.append(
            TileInfo(
                tile_index=len(tiles),
                path=full_path,
                bbox=(0, 0, width, height),
                row=-1,
                col=-1,
                source_image=image_path,
                is_full_image=True,
            )
        )

    for r in range(effective_rows):
        for c in range(effective_cols):
            left = max(0, c * tile_width - overlap)
            upper = max(0, r * tile_height - overlap)
            right = min(width, (c + 1) * tile_width + overlap)
            lower = min(height, (r + 1) * tile_height + overlap)
            box = (left, upper, right, lower)
            tile = image.crop(box)
            tile_index = len(tiles)
            out_path = output_dir / f"tile_{tile_index:02d}_{left}_{upper}_{right}_{lower}.png"
            tile.save(out_path, "PNG", optimize=True)
            tiles.append(
                TileInfo(
                    tile_index=tile_index,
                    path=out_path,
                    bbox=box,
                    row=r,
                    col=c,
                    source_image=image_path,
                    is_full_image=False,
                )
            )

    return tiles


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split CAD images into overlapping tiles.")
    parser.add_argument("image", type=Path, help="Path to the source PNG image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tiles_out"),
        help="Directory to store generated tiles.",
    )
    parser.add_argument("--rows", type=int, default=3, help="Number of tile rows (default: 3).")
    parser.add_argument("--cols", type=int, default=4, help="Number of tile columns (default: 4).")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap pixels between tiles (default: 200).")
    parser.add_argument("--trim", action="store_true", help="Enable white border trimming.")
    parser.add_argument(
        "--trim-tol",
        type=int,
        default=245,
        help="Higher values trim more aggressively (default: 245).",
    )
    parser.add_argument("--enhance", action="store_true", help="Enable contrast/sharpness enhancements.")
    parser.add_argument(
        "--exclude-full",
        action="store_true",
        help="Do not save the full page image as an additional tile.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tiles = split_image_into_tiles(
        args.image,
        args.output_dir,
        rows=args.rows,
        cols=args.cols,
        overlap=args.overlap,
        trim=args.trim,
        trim_tol=args.trim_tol,
        enhance=args.enhance,
        include_full_image=not args.exclude_full,
    )
    print(f"生成 {len(tiles)} 个图块: {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
