#!/usr/bin/env python3
"""Gradio front-end for the multimodal CAD analysis helper."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import gradio as gr
from PIL import Image

from multimodal_prompt import stream_modelscope_endpoint
from pdf2png import convert_pdf_to_pngs
from split_connectivity import graph_split
from split_img import TileInfo, split_image_into_tiles


DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_BASE_URL = os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1"
DEFAULT_SYSTEM_PROMPT = "你是一个CAD工程图纸识别的助手，你的任务是分析给定的图纸，给出专业的数值。"
SCROLL_CSS = """
.scroll-output textarea {
    min-height: 320px !important;
    max-height: 540px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
}
"""

PDF_DPI = int(os.getenv("CAD_REG_PDF_DPI", "350"))
TILING_ROWS = int(os.getenv("CAD_REG_TILE_ROWS", "3"))
TILING_COLS = int(os.getenv("CAD_REG_TILE_COLS", "4"))
TILING_OVERLAP = int(os.getenv("CAD_REG_TILE_OVERLAP", "200"))
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAGE_OUTPUT_DIR = Path(os.getenv("CAD_REG_PAGE_DIR", "out_png"))
TILE_OUTPUT_DIR = Path(os.getenv("CAD_REG_TILE_DIR", "tiles_out"))
CONNECTIVITY_MERGE_THRESHOLD = float(os.getenv("CAD_REG_CONN_MERGE_THRESHOLD", "11.5"))
CONNECTIVITY_GRID_SIZE = int(os.getenv("CAD_REG_CONN_GRID_SIZE", "200"))
CONNECTIVITY_MAX_AREA_RIOT = float(os.getenv("CAD_REG_CONN_MAX_AREA_RIOT", "0.25"))
CONNECTIVITY_MIN_AREA = int(os.getenv("CAD_REG_CONN_MIN_AREA", "150"))


@dataclass
class TileContext:
    order: int
    source_file: str
    page_number: int
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    path: Path
    is_full_image: bool


@dataclass
class PageImage:
    source_file: str
    page_number: int
    path: Path
    doc_slug: str
    width: int
    height: int


def _resolve_file_paths(files: Sequence[object]) -> List[Path]:
    paths: List[Path] = []
    for file in files:
        if file is None:
            continue
        if isinstance(file, str):
            paths.append(Path(file))
        elif hasattr(file, "name"):
            paths.append(Path(file.name))
        elif isinstance(file, dict) and "name" in file:
            paths.append(Path(str(file["name"])))
    return paths


def _default_api_key() -> str | None:
    return (
        os.getenv("MODELSCOPE_API_KEY")
        or os.getenv("MODELSCOPE_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _prepare_run_directories(run_id: str) -> tuple[Path, Path]:
    pages_root = PAGE_OUTPUT_DIR / run_id
    tiles_root = TILE_OUTPUT_DIR / run_id
    pages_root.mkdir(parents=True, exist_ok=True)
    tiles_root.mkdir(parents=True, exist_ok=True)
    return pages_root, tiles_root


def _prepare_page_images(paths: Sequence[Path], pages_root: Path) -> List[PageImage]:
    page_images: List[PageImage] = []
    for idx, src_path in enumerate(paths):
        doc_slug = f"doc_{idx:03d}_{src_path.stem}"
        source_name = src_path.name
        if _is_pdf(src_path):
            pdf_dir = pages_root / doc_slug
            page_paths = convert_pdf_to_pngs(
                src_path,
                pdf_dir,
                dpi=PDF_DPI,
                trim=False,
                trim_tol=245,
                prefix=doc_slug,
            )
        elif _is_supported_image(src_path):
            doc_dir = pages_root / doc_slug
            doc_dir.mkdir(parents=True, exist_ok=True)
            dest_image = doc_dir / source_name
            shutil.copy2(src_path, dest_image)
            page_paths = [dest_image]
        else:
            raise ValueError(f"暂不支持的文件类型: {src_path.name}")

        if not page_paths:
            raise ValueError(f"{src_path.name} 未能生成有效的页面图像。")

        for page_idx, page_path in enumerate(page_paths, start=1):
            with Image.open(page_path) as img:
                width, height = img.size
            page_images.append(
                PageImage(
                    source_file=source_name,
                    page_number=page_idx,
                    path=page_path,
                    doc_slug=doc_slug,
                    width=width,
                    height=height,
                )
            )
    return page_images


def _prepare_grid_tiles(page_images: Sequence[PageImage], tiles_root: Path) -> List[TileContext]:
    contexts: List[TileContext] = []
    tile_counter = 0
    for page in page_images:
        tile_dir = tiles_root / f"{page.doc_slug}_page_{page.page_number:03d}"
        tiles: List[TileInfo] = split_image_into_tiles(
            page.path,
            tile_dir,
            rows=TILING_ROWS,
            cols=TILING_COLS,
            overlap=TILING_OVERLAP,
            trim=False,
            trim_tol=245,
            enhance=False,
            include_full_image=True,
        )
        if not tiles:
            raise ValueError(f"{page.source_file} 第 {page.page_number} 页未生成任何图块。")
        for tile in tiles:
            contexts.append(
                TileContext(
                    order=tile_counter,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    row=tile.row,
                    col=tile.col,
                    bbox=tile.bbox,
                    path=tile.path,
                    is_full_image=tile.is_full_image,
                )
            )
            tile_counter += 1
    return contexts


def _prepare_connectivity_tiles(page_images: Sequence[PageImage], tiles_root: Path) -> List[TileContext]:
    contexts: List[TileContext] = []
    tile_counter = 0
    for page in page_images:
        tile_dir = tiles_root / f"{page.doc_slug}_page_{page.page_number:03d}_conn"
        tile_dir.mkdir(parents=True, exist_ok=True)
        segments = graph_split(
            str(page.path),
            tile_dir.as_posix(),
            merge_threshold=CONNECTIVITY_MERGE_THRESHOLD,
            grid_size=CONNECTIVITY_GRID_SIZE,
            max_area_riot=CONNECTIVITY_MAX_AREA_RIOT,
            min_area=CONNECTIVITY_MIN_AREA,
        )
        if not segments:
            continue
        for segment_path, bbox in segments:
            x1, y1, x2, y2 = bbox
            contexts.append(
                TileContext(
                    order=tile_counter,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    row=-1,
                    col=-1,
                    bbox=(x1, y1, x2, y2),
                    path=Path(segment_path),
                    is_full_image=False,
                )
            )
            tile_counter += 1
    return contexts


def _parse_split_choice(choice: str | None) -> bool:
    if choice is None:
        return True
    normalized = choice.strip().lower()
    if normalized in {"切块识别", "切块", "split"}:
        return True
    if normalized in {"原图识别", "原图", "nosplit"}:
        return False
    return True


def _resolve_split_strategy(choice: str | None) -> str:
    if not choice:
        return "网格分块"
    choice = choice.strip()
    if choice in {"网格分块", "连通域分块"}:
        return choice
    return "网格分块"


def _format_tile_context(tile_contexts: Sequence[TileContext]) -> str:
    lines = []
    for tile in tile_contexts:
        position_desc = (
            "全图"
            if tile.is_full_image
            else (
                f"行: {tile.row + 1} | 列: {tile.col + 1}"
                if tile.row >= 0 and tile.col >= 0
                else "自适应块"
            )
        )
        lines.append(
            (
                f"[块{tile.order + 1:02d}] 文件: {tile.source_file} | 页: {tile.page_number} | "
                f"{position_desc} | 区域: {tile.bbox}"
            )
        )
    return "图块上下文（按顺序送入模型）：\n" + "\n".join(lines)


def analyze_cad_images(
    files: Sequence[object],
    prompt: str,
    extra_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    split_choice: str,
    split_strategy: str,
):
    if not files:
        yield "请至少上传一张图纸。"
        return
    paths = _resolve_file_paths(files)
    if not paths:
        yield "未能解析上传的图纸文件。"
        return
    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        yield f"以下文件不存在或不可用: {missing_str}"
        return

    final_prompt = (prompt or "").strip()
    if extra_prompt and extra_prompt.strip():
        final_prompt = f"{final_prompt}\n\n补充说明：{extra_prompt.strip()}"
    if not final_prompt:
        yield "请输入提示词。"
        return

    resolved_key = api_key.strip() or _default_api_key()
    if not resolved_key:
        yield "请提供有效的 ModelScope API Key。"
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        pages_root, tiles_root = _prepare_run_directories(run_id)
    except Exception as exc:
        yield f"无法准备输出目录: {exc}"
        return

    should_split = _parse_split_choice(split_choice)
    strategy_choice = _resolve_split_strategy(split_strategy)

    try:
        page_images = _prepare_page_images(paths, pages_root)
    except Exception as exc:
        yield f"预处理失败: {exc}"
        return

    if should_split:
        try:
            if strategy_choice == "连通域分块":
                tile_contexts = _prepare_connectivity_tiles(page_images, tiles_root)
            else:
                tile_contexts = _prepare_grid_tiles(page_images, tiles_root)
        except Exception as exc:
            yield f"分块失败: {exc}"
            return
        if not tile_contexts:
            yield "预处理失败：未生成任何图块。"
            return
        artifact_desc = f"图块已保存至 {tiles_root.resolve()}。"
    else:
        tile_contexts = []
        for order, page in enumerate(page_images):
            tile_contexts.append(
                TileContext(
                    order=order,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    row=-1,
                    col=-1,
                    bbox=(0, 0, page.width, page.height),
                    path=page.path,
                    is_full_image=True,
                )
            )
        if not tile_contexts:
            yield "预处理失败：未生成任何原图供识别。"
            return
        artifact_desc = f"原图已保存至 {pages_root.resolve()}。"

    tile_paths = [ctx.path for ctx in tile_contexts]
    tile_context_text = _format_tile_context(tile_contexts)
    final_prompt = f"{final_prompt}\n\n{tile_context_text}"

    mode_desc = "切块识别" if should_split else "原图识别"
    strategy_desc = f"切块方式: {strategy_choice}" if should_split else "使用原图"
    aggregated = (
        f"{mode_desc}预处理完成：共准备 {len(tile_paths)} 个输入图像，保持顺序用于上下文。"
        f"{artifact_desc}（批次: {run_id}; {strategy_desc}）\n\n"
    )
    yield aggregated

    received_tokens = False
    try:
        for delta in stream_modelscope_endpoint(
            base_url=base_url.strip() or DEFAULT_BASE_URL,
            api_key=resolved_key,
            model=model.strip() or DEFAULT_MODEL,
            system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
            prompt=final_prompt,
            image_paths=tile_paths,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        ):
            received_tokens = True
            aggregated += delta
            yield aggregated
    except Exception as exc:  # pragma: no cover - surface runtime errors in UI
        message = (aggregated + f"\n\n[错误] {exc}") if aggregated else f"推理失败: {exc}"
        yield message
        return

    if not received_tokens:
        aggregated = aggregated.rstrip() + "\n\n模型未返回有效文本，请重试。"
        yield aggregated


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="CAD 图纸识别助手", css=SCROLL_CSS) as demo:
        gr.Markdown("## CAD 工程图纸识别助手\n上传多张 CAD 图纸并输入要分析的问题。")

        with gr.Row():
            file_input = gr.File(
                label="CAD 图纸 (可多张)",
                file_types=["image", "application/pdf"],
                file_count="multiple",
                type="file",
            )
            split_mode = gr.Radio(
                label="处理方式",
                choices=["切块识别", "原图识别"],
                value="切块识别",
            )
            split_strategy = gr.Radio(
                label="切块方式（仅在切块识别时生效）",
                choices=["网格分块", "连通域分块"],
                value="网格分块",
            )

        output_box = gr.Textbox(
            label="分析结果",
            lines=16,
            interactive=False,
            elem_classes=["scroll-output"],
        )

        prompt_box = gr.Textbox(
            label="用户提示词",
            placeholder="例如：综合分析两张楼层图的差异并给出关键尺寸。",
            lines=3,
        )

        extra_prompt_box = gr.Textbox(
            label="补充提示词（可选）",
            placeholder="例如：请重点关注结构标高和楼层关系。",
            lines=2,
        )

        system_box = gr.Textbox(
            label="系统提示词",
            value=DEFAULT_SYSTEM_PROMPT,
            lines=2,
        )

        with gr.Accordion("高级设置", open=False):
            model_box = gr.Textbox(label="模型 ID", value=DEFAULT_MODEL)
            base_url_box = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL)
            api_key_box = gr.Textbox(
                label="ModelScope API Key",
                value=_default_api_key() or "",
                type="password",
            )
            temperature_slider = gr.Slider(
                label="Temperature",
                value=0.2,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            )
            max_tokens_slider = gr.Slider(
                label="Max Tokens",
                value=2048,
                minimum=256,
                maximum=4096,
                step=128,
            )

        run_button = gr.Button("分析图纸", variant="primary")
        run_button.click(
            analyze_cad_images,
            inputs=[
                file_input,
                prompt_box,
                extra_prompt_box,
                system_box,
                model_box,
                base_url_box,
                api_key_box,
                temperature_slider,
                max_tokens_slider,
                split_mode,
                split_strategy,
            ],
            outputs=output_box,
        )
    return demo


def _parse_bool_env(value: str | None) -> tuple[bool, bool]:
    if value is None:
        return False, False
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True, True
    if lowered in {"0", "false", "no", "off"}:
        return True, False
    return False, False


def launch_app() -> None:
    demo = build_interface().queue()
    share_env = os.getenv("GRADIO_SHARE")
    share_explicit, share_value = _parse_bool_env(share_env)
    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("GRADIO_SERVER_PORT")
    port_value = int(server_port) if server_port and server_port.isdigit() else None

    launch_kwargs = {
        "share": share_value if share_explicit else False,
        "server_name": server_name,
        "server_port": port_value,
        "show_api": False,
    }

    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "share=True" in str(exc) and not share_explicit:
            demo.launch(
                share=True,
                server_name=server_name,
                server_port=port_value,
                show_api=False,
            )
        else:
            raise


if __name__ == "__main__":
    launch_app()
