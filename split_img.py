#!/usr/bin/env python3
"""Split high-resolution CAD drawings into overlapping image tiles."""

from __future__ import annotations

import argparse
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
    include_full_image: bool = True,
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
    tile_width = width // cols
    tile_height = height // rows
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

    for r in range(rows):
        for c in range(cols):
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
