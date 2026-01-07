#!/usr/bin/env python3
"""Gradio front-end for the multimodal CAD analysis helper."""

from __future__ import annotations

import os
import shutil
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import gradio as gr
from PIL import Image

from embedding_utils import embed_texts
from milvus_store import insert_blocks, search_blocks
from multimodal_prompt import (
    call_modelscope_endpoint,
    stream_modelscope_endpoint,
)
from pdf2png import convert_pdf_to_pngs
from split_connectivity import graph_split
from split_img import TileInfo, split_image_into_tiles


DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_BASE_URL = os.getenv("MODELSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBED_MODEL = os.getenv("CAD_RAG_EMBED_MODEL", "text-embedding-v4")
DEFAULT_EMBED_BASE_URL = os.getenv("CAD_RAG_EMBED_BASE_URL") or DEFAULT_BASE_URL
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
CONNECTIVITY_OUTPUT_DIR = Path(os.getenv("CAD_REG_CONN_DIR", "conn_blocks"))
CONNECTIVITY_MERGE_THRESHOLD = float(os.getenv("CAD_REG_CONN_MERGE_THRESHOLD", "11.5"))
CONNECTIVITY_GRID_SIZE = int(os.getenv("CAD_REG_CONN_GRID_SIZE", "200"))
CONNECTIVITY_MAX_AREA_RIOT = float(os.getenv("CAD_REG_CONN_MAX_AREA_RIOT", "0.25"))
CONNECTIVITY_MIN_AREA = int(os.getenv("CAD_REG_CONN_MIN_AREA", "150"))
GRID_MAX_TILE_DIM = int(os.getenv("CAD_REG_GRID_MAX_TILE_DIM", "1000"))
DEFAULT_BLOCK_SUMMARY_PROMPT = os.getenv(
    "CAD_REG_BLOCK_PROMPT",
    "请用 1-2 句话描述此 CAD 图块涵盖的结构/构件、尺寸或关键标注，并指出可见的标签。",
)
DEFAULT_RAG_TOP_K = int(os.getenv("CAD_REG_RAG_TOP_K", "3"))


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


@dataclass
class BlockHit:
    block_id: str
    doc_id: str
    page_number: int
    bbox: tuple[int, int, int, int]
    image_path: Path
    summary: str
    score: float


def _block_hit_from_record(record: dict) -> BlockHit | None:
    image_path = Path(record.get("image_path", ""))
    if not image_path.exists():
        return None
    bbox = _parse_bbox(record.get("bbox"))
    return BlockHit(
        block_id=str(record.get("block_id")),
        doc_id=str(record.get("doc_id", "")),
        page_number=int(record.get("page_number", 0)),
        bbox=bbox,
        image_path=image_path,
        summary=str(record.get("summary", "")),
        score=float(record.get("score", 0.0)),
    )


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


def _default_embedding_api_key() -> str | None:
    return (
        os.getenv("CAD_RAG_EMBED_API_KEY")
        or os.getenv("MODELSCOPE_EMBED_API_KEY")
        or _default_api_key()
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


def _prepare_ingest_directories(run_id: str) -> tuple[Path, Path]:
    pages_root = PAGE_OUTPUT_DIR / f"kb_{run_id}"
    conn_root = CONNECTIVITY_OUTPUT_DIR / f"kb_{run_id}"
    pages_root.mkdir(parents=True, exist_ok=True)
    conn_root.mkdir(parents=True, exist_ok=True)
    return pages_root, conn_root


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


def _compute_grid_slices(width: int, height: int) -> tuple[int, int]:
    rows = max(1, math.ceil(height / GRID_MAX_TILE_DIM))
    cols = max(1, math.ceil(width / GRID_MAX_TILE_DIM))
    return rows, cols


def _parse_bbox(value: str | None) -> tuple[int, int, int, int]:
    if not value:
        return (0, 0, 0, 0)
    try:
        parts = [int(float(x.strip())) for x in value.split(",")]
        while len(parts) < 4:
            parts.append(0)
        return tuple(parts[:4])  # type: ignore[return-value]
    except Exception:
        return (0, 0, 0, 0)


def _describe_tile(tile: TileContext) -> str:
    position_desc = (
        "全图"
        if tile.is_full_image
        else (
            f"行: {tile.row + 1} | 列: {tile.col + 1}"
            if tile.row >= 0 and tile.col >= 0
            else "自适应块"
        )
    )
    return (
        f"[块{tile.order + 1:02d}] 文件: {tile.source_file} | 页: {tile.page_number} | "
        f"{position_desc} | 区域: {tile.bbox}"
    )


def _build_block_prompt(base_prompt: str, tile: TileContext) -> str:
    desc = _describe_tile(tile)
    instructions = (
        "请仅根据上述图块提供的内容回答用户问题相关的信息，"
        "列出你在该块中发现的关键尺寸、标注或异常，若信息不足请说明。"
    )
    components = [base_prompt.strip(), "当前处理的图块：", desc, instructions]
    return "\n\n".join(component for component in components if component)


def _build_summary_prompt(base_prompt: str, blocks: Sequence[tuple[TileContext, str]]) -> str:
    lines = [base_prompt.strip(), "以下是针对每个图块得到的识别结果："]
    for tile, text in blocks:
        lines.append(f"{_describe_tile(tile)}\n{text.strip()}")
    lines.append(
        "请综合所有图块的信息，给出最终答案，必要时指出块之间的矛盾或补充说明。"
    )
    return "\n\n".join(line for line in lines if line.strip())






def rag_answer(
    files: Sequence[object],
    summary_prompt: str,
    question: str,
    extra_prompt: str,
    top_k: int,
    embed_model: str,
    embed_base_url: str,
    embed_api_key: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
):
    def _initial_output(
        message: str,
    ) -> tuple[str, list[list[str]], list[list[str]], list[list[str]]]:
        return message, [], [], []

    paths = _resolve_file_paths(files)
    if not paths:
        yield _initial_output("请先上传 CAD 图纸（PDF 或 PNG）。")
        return
    missing = [p for p in paths if not p.exists()]
    if missing:
        yield _initial_output(f"以下文件不存在或不可用: {', '.join(str(p) for p in missing)}")
        return

    base_prompt = (question or "").strip()
    if not base_prompt:
        yield _initial_output("请输入要查询的问题。")
        return

    describe_prompt = summary_prompt.strip() or DEFAULT_BLOCK_SUMMARY_PROMPT
    resolved_key = api_key.strip() or _default_api_key()
    if not resolved_key:
        yield _initial_output("请提供有效的 ModelScope API Key。")
        return

    embed_model_value = embed_model.strip() if embed_model and embed_model.strip() else None
    embed_base_value = embed_base_url.strip() if embed_base_url and embed_base_url.strip() else None
    embed_key_value = embed_api_key.strip() if embed_api_key and embed_api_key.strip() else None
    embed_kwargs = {
        "model": embed_model_value,
        "base_url": embed_base_value,
        "api_key": embed_key_value,
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pages_root, conn_root = _prepare_ingest_directories(run_id)
    try:
        page_images = _prepare_page_images(paths, pages_root)
    except Exception as exc:
        yield _initial_output(f"预处理失败: {exc}")
        return
    if not page_images:
        yield _initial_output("未生成任何页面，无法构建知识库。")
        return

    aggregated = f"[{run_id}] 开始构建临时知识库...\n"
    candidate_gallery: List[list[str]] = []
    grid_gallery: List[list[str]] = []
    connectivity_gallery: List[list[str]] = []

    def current_state():
        return (
            aggregated,
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )

    yield current_state()

    records: List[dict] = []
    doc_ids: set[str] = set()

    for page in page_images:
        doc_id = f"{page.doc_slug}_{run_id}"
        doc_ids.add(doc_id)
        aggregated += f"处理 {page.source_file} 第 {page.page_number} 页...\n"
        yield current_state()

        conn_dir = conn_root / f"{page.doc_slug}_page_{page.page_number:03d}"
        segments = graph_split(
            str(page.path),
            conn_dir.as_posix(),
            merge_threshold=CONNECTIVITY_MERGE_THRESHOLD,
            grid_size=CONNECTIVITY_GRID_SIZE,
            max_area_riot=CONNECTIVITY_MAX_AREA_RIOT,
            min_area=CONNECTIVITY_MIN_AREA,
        )
        if not segments:
            aggregated += "  未检测到连通域块。\n"
            yield current_state()
            continue

        for idx, (segment_path, bbox) in enumerate(segments):
            block_id = f"{doc_id}_b{idx:03d}"
            aggregated += f"  连通域块 {block_id} 提取语义...\n"
            segment_path_obj = Path(segment_path)
            if segment_path_obj.exists():
                connectivity_gallery.append(
                    [
                        str(segment_path_obj),
                        (
                            f"{page.source_file} 页{page.page_number} "
                            f"块{idx + 1:03d} 区域{bbox}"
                        ),
                    ]
                )
            yield current_state()
            prompt_text = (
                f"{describe_prompt}\n\n文件: {page.source_file} 页码: {page.page_number}。"
                "请聚焦该图块本身的结构、标注与尺寸。"
            )
            try:
                description = call_modelscope_endpoint(
                    base_url=base_url.strip() or DEFAULT_BASE_URL,
                    api_key=resolved_key,
                    model=model.strip() or DEFAULT_MODEL,
                    system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
                    prompt=prompt_text,
                    image_paths=[segment_path_obj],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                if not description:
                    description = "模型未返回描述。"
            except Exception as exc:
                description = f"描述失败: {exc}"
            records.append(
                {
                    "block_id": block_id,
                    "doc_id": doc_id,
                    "page_number": page.page_number,
                    "image_path": str(segment_path_obj.resolve()),
                    "bbox": bbox,
                    "summary": description,
                }
            )
            preview = description.strip().replace("\n", " ")
            aggregated += f"    完成。语义：{preview}\n"
            yield current_state()

    if not records:
        yield (
            "未检测到任何连通域块，请检查图纸。",
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )
        return

    try:
        embeddings = embed_texts([record["summary"] for record in records], **embed_kwargs)
    except Exception as exc:
        yield (
            f"生成嵌入失败: {exc}",
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )
        return

    try:
        insert_blocks(records, embeddings)
        aggregated += f"知识库构建完成：写入 {len(records)} 个块。\n"
        yield current_state()
    except Exception as exc:
        yield (
            f"写入 Milvus 失败: {exc}",
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )
        return

    try:
        query_vector = embed_texts([base_prompt], **embed_kwargs)[0]
    except Exception as exc:
        yield (
            f"生成查询向量失败: {exc}",
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )
        return

    doc_id_list = sorted(doc_ids)
    rag_top_k = int(top_k) if top_k else DEFAULT_RAG_TOP_K
    raw_hits = search_blocks(
        query_vector,
        top_k=rag_top_k,
        doc_ids=doc_id_list,
    )
    block_hits = [hit for record in raw_hits if (hit := _block_hit_from_record(record))]
    if not block_hits:
        yield (
            "未检索到匹配的连通域块，请尝试其他问题。",
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )
        return

    _, tiles_root = _prepare_run_directories(f"rag_{run_id}")

    block_sizes: dict[str, tuple[int, int]] = {}
    aggregated += "检索到以下候选连通域块：\n"
    for idx, hit in enumerate(block_hits, start=1):
        aggregated += (
            f"[候选{idx}] doc={hit.doc_id} page={hit.page_number} score={hit.score:.3f}\n"
            f"摘要：{hit.summary}\n\n"
        )
        size_suffix = ""
        if hit.image_path.exists():
            try:
                with Image.open(hit.image_path) as preview_img:
                    width, height = preview_img.size
                block_sizes[hit.block_id] = (width, height)
                size_suffix = f" | {width}x{height}px"
            except Exception as exc:
                aggregated += f"[候选{idx}] 无法读取图块尺寸: {exc}\n"
        if hit.image_path.exists():
            candidate_gallery.append(
                [str(hit.image_path), f"候选{idx} 分数 {hit.score:.3f}{size_suffix}"]
            )
    yield current_state()

    all_tile_results: List[tuple[TileContext, str]] = []
    order_counter = 0

    for rank, hit in enumerate(block_hits, start=1):
        block_path = hit.image_path
        if not block_path.exists():
            aggregated += f"[候选{rank}] 图块文件缺失：{block_path}\n"
            yield current_state()
            continue

        width: int | None = None
        height: int | None = None
        cached_size = block_sizes.get(hit.block_id)
        if cached_size:
            width, height = cached_size
        else:
            try:
                with Image.open(block_path) as img:
                    width, height = img.size
                block_sizes[hit.block_id] = (width, height)
            except Exception as exc:
                aggregated += f"[候选{rank}] 无法打开图块: {exc}\n"
                yield current_state()
                continue
        if width is None or height is None:
            aggregated += f"[候选{rank}] 图块尺寸缺失，跳过。\n"
            yield current_state()
            continue

        if width <= 1200 and height <= 1200:
            tiles = [
                TileContext(
                    order=order_counter,
                    source_file=hit.doc_id,
                    page_number=hit.page_number,
                    row=-1,
                    col=-1,
                    bbox=hit.bbox,
                    path=block_path,
                    is_full_image=True,
                )
            ]
            order_counter += 1
            aggregated += f"[候选{rank}] 图块尺寸 {width}x{height}，直接整体识别。\n"
            if block_path.exists():
                label = f"{hit.doc_id} 全图 {width}x{height}px"
                grid_gallery.append([str(block_path), label])
            yield current_state()
        else:
            rows, cols = _compute_grid_slices(width, height)
            block_dir = tiles_root / hit.block_id
            raw_tiles = split_image_into_tiles(
                block_path,
                block_dir,
                rows=rows,
                cols=cols,
                overlap=TILING_OVERLAP,
                trim=False,
                trim_tol=245,
                enhance=False,
                include_full_image=False,
            )
            if not raw_tiles:
                aggregated += f"[候选{rank}] 未能生成网格切片。\n"
                yield current_state()
                continue

            tiles = []
            for tile in raw_tiles:
                tiles.append(
                    TileContext(
                        order=order_counter,
                        source_file=hit.doc_id,
                        page_number=hit.page_number,
                        row=tile.row,
                        col=tile.col,
                        bbox=tile.bbox,
                        path=tile.path,
                        is_full_image=tile.is_full_image,
                    )
                )
                if tile.path.exists():
                    tile_w = max(1, tile.bbox[2] - tile.bbox[0])
                    tile_h = max(1, tile.bbox[3] - tile.bbox[1])
                    grid_gallery.append(
                        [
                            str(tile.path),
                            f"{hit.doc_id} 块{order_counter:03d} {tile_w}x{tile_h}px",
                        ]
                    )
                order_counter += 1

            aggregated += f"[候选{rank}] 生成 {len(tiles)} 个网格切片。\n"
            yield current_state()

        for tile_ctx in tiles:
            tile_label = f"[候选{rank}-块{tile_ctx.order:03d}]"
            tile_header = f"{tile_label} 结果：\n"
            tile_output = ""
            yield (
                aggregated + tile_header,
                list(candidate_gallery),
                list(grid_gallery),
                list(connectivity_gallery),
            )

            combined_prompt = base_prompt
            if extra_prompt and extra_prompt.strip():
                combined_prompt = f"{combined_prompt}\n\n补充说明：{extra_prompt.strip()}"
            block_prompt = _build_block_prompt(combined_prompt, tile_ctx)

            try:
                received = False
                for delta in stream_modelscope_endpoint(
                    base_url=base_url.strip() or DEFAULT_BASE_URL,
                    api_key=resolved_key,
                    model=model.strip() or DEFAULT_MODEL,
                    system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
                    prompt=block_prompt,
                    image_paths=[tile_ctx.path],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                ):
                    received = True
                    tile_output += delta
                    yield (
                        aggregated + tile_header + tile_output,
                        list(candidate_gallery),
                        list(grid_gallery),
                        list(connectivity_gallery),
                    )
                if not received:
                    tile_output = "模型未返回有效文本。"
            except Exception as exc:
                tile_output = f"识别失败: {exc}"
                yield (
                    aggregated + tile_header + tile_output,
                    list(candidate_gallery),
                    list(grid_gallery),
                    list(connectivity_gallery),
                )

            all_tile_results.append((tile_ctx, tile_output))
            aggregated += tile_header + tile_output + "\n"
            yield current_state()

    aggregated += "候选块分析完成，正在整合最终答案...\n"
    yield current_state()

    combined_summary_prompt = base_prompt
    if extra_prompt and extra_prompt.strip():
        combined_summary_prompt = f"{combined_summary_prompt}\n\n补充说明：{extra_prompt.strip()}"
    summary_prompt = _build_summary_prompt(combined_summary_prompt, all_tile_results)
    summary_header = "综合回答：\n"
    summary_output = ""
    yield (
        aggregated + summary_header,
        list(candidate_gallery),
        list(grid_gallery),
        list(connectivity_gallery),
    )

    try:
        received = False
        for delta in stream_modelscope_endpoint(
            base_url=base_url.strip() or DEFAULT_BASE_URL,
            api_key=resolved_key,
            model=model.strip() or DEFAULT_MODEL,
            system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
            prompt=summary_prompt,
            image_paths=[],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        ):
            received = True
            summary_output += delta
            yield (
                aggregated + summary_header + summary_output,
                list(candidate_gallery),
                list(grid_gallery),
                list(connectivity_gallery),
            )
        if not received:
            summary_output = "模型未返回有效文本。"
    except Exception as exc:
        summary_output = f"汇总失败: {exc}"
        yield (
            aggregated + summary_header + summary_output,
            list(candidate_gallery),
            list(grid_gallery),
            list(connectivity_gallery),
        )
        return

    aggregated += summary_header + summary_output
    yield current_state()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="CAD 图纸识别助手", css=SCROLL_CSS) as demo:
        gr.Markdown("## CAD 图纸识别助手（上传 PDF → 连通域分块 → RAG 问答）")

        with gr.Accordion("模型与推理配置", open=False):
            system_box = gr.Textbox(
                label="系统提示词",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=2,
            )
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
            gr.Markdown("### 嵌入模型配置")
            embed_model_box = gr.Textbox(
                label="Embedding 模型 ID",
                value=DEFAULT_EMBED_MODEL,
            )
            embed_base_url_box = gr.Textbox(
                label="Embedding Base URL",
                value=DEFAULT_EMBED_BASE_URL,
            )
            embed_api_key_box = gr.Textbox(
                label="Embedding API Key",
                value=_default_embedding_api_key() or "",
                type="password",
            )

        file_input = gr.File(
            label="CAD 图纸 (PDF/PNG，可多文件)",
            file_types=["image", "application/pdf"],
            file_count="multiple",
            type="file",
        )
        summary_prompt_box = gr.Textbox(
            label="连通域语义提取提示词",
            value=DEFAULT_BLOCK_SUMMARY_PROMPT,
            lines=3,
        )
        question_box = gr.Textbox(
            label="用户问题",
            placeholder="例如：请找出图纸中剪力墙编号 WZ 的尺寸和标高。",
            lines=3,
        )
        extra_prompt_box = gr.Textbox(
            label="补充提示词（可选）",
            placeholder="例如：若存在冲突请列出不同来源。",
            lines=2,
        )
        topk_slider = gr.Slider(
            label="RAG 候选块数量",
            minimum=1,
            maximum=10,
            step=1,
            value=DEFAULT_RAG_TOP_K,
        )
        answer_box = gr.Textbox(
            label="答案（流式输出）",
            lines=20,
            interactive=False,
            elem_classes=["scroll-output"],
        )
        gr.Markdown("### 连通域分块预览（全部图块）")
        connectivity_gallery_box = gr.Gallery(
            label=None,
            columns=4,
            height="auto",
            interactive=False,
        )

        gr.Markdown("### 图块预览")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**候选连通域块**")
                candidate_gallery_box = gr.Gallery(
                    label=None,
                    columns=3,
                    height="auto",
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("**网格分块（命中块）**")
                grid_gallery_box = gr.Gallery(
                    label=None,
                    columns=3,
                    height="auto",
                    interactive=False,
                )
        answer_button = gr.Button("上传并开始问答", variant="primary")
        answer_button.click(
            rag_answer,
            inputs=[
                file_input,
                summary_prompt_box,
                question_box,
                extra_prompt_box,
                topk_slider,
                embed_model_box,
                embed_base_url_box,
                embed_api_key_box,
                system_box,
                model_box,
                base_url_box,
                api_key_box,
                temperature_slider,
                max_tokens_slider,
            ],
            outputs=[
                answer_box,
                candidate_gallery_box,
                grid_gallery_box,
                connectivity_gallery_box,
            ],
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
