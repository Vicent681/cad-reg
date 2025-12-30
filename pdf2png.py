#!/usr/bin/env python3
"""Utility functions/CLI to convert PDF CAD drawings into trimmed PNG pages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from pdf2image import convert_from_path
from PIL import Image, ImageChops


def trim_white_border(im: Image.Image, tol: int = 245) -> Image.Image:
    """Trim near-white border pixels from an image."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    diff = diff.point(lambda p: 0 if p > (255 - tol) else 255)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


def convert_pdf_to_pngs(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 400,
    trim: bool = True,
    trim_tol: int = 245,
    prefix: str | None = None,
) -> List[Path]:
    """
    Convert a PDF into per-page PNG files and optionally trim borders.

    Returns a list of generated PNG paths ordered by page number.
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    generated: List[Path] = []
    prefix = prefix or pdf_path.stem

    for index, page in enumerate(pages, start=1):
        processed = trim_white_border(page, tol=trim_tol) if trim else page
        out_path = output_dir / f"{prefix}_page_{index:03d}.png"
        processed.save(out_path, "PNG", optimize=True)
        generated.append(out_path)

    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CAD PDFs to PNG pages.")
    parser.add_argument("pdf", type=Path, help="Path to the source PDF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out_png"),
        help="Directory to store generated PNG pages.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI (default: 400).")
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Skip trimming near-white borders around each page.",
    )
    parser.add_argument(
        "--trim-tol",
        type=int,
        default=245,
        help="Higher values trim more aggressively (default: 245).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pngs = convert_pdf_to_pngs(
        args.pdf,
        args.output_dir,
        dpi=args.dpi,
        trim=not args.no_trim,
        trim_tol=args.trim_tol,
    )
    print(f"生成 {len(pngs)} 张 PNG 图: {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
